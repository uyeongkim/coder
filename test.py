import numpy as np
import random
import os
import sys
from pathlib import Path
import re
from tqdm import tqdm, trange
from multiprocessing import cpu_count
from typing import Any, Dict, List, Tuple, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from datasets import load_dataset
import logging
import concurrent.futures
from functools import partial
import csv
import multiprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_qwen_model(model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Optimized quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True  # Better memory management
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # Reduce CPU memory usage
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Use Flash Attention for speed
    )
    
    # Optimize for inference
    model.eval()
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Model {model_name} loaded successfully.")
    return tokenizer, model

class CodeContestDataset:
    def __init__(self, split: str = "train", max_problems: int = -1, max_cases: int = -1):
        def _is_valid(row):
            if row is None:
                return False
            if row.get('input_file') or row.get('output_file'):
                return False
            if row.get('description') is None or not (10 < len(row['description']) < 1500):
                return False
            if row.get('name') is None or row.get('private_tests') is None or len(row['private_tests']['input']) == 0:
                return False
            if row.get('solutions') is None or 1 not in row['solutions']['language'] or not any(len(sol.strip()) < 500 for sol in row['solutions']['solution']):
                return False
            if row.get('time_limit') is None or not (row['time_limit'].get('seconds') or row['time_limit'].get('nanos')):
                return False
            if row.get('cf_rating') is None:
                return False
            return True
        
        if max_problems <= 0: max_problems = int(sys.maxsize)
        if max_cases <= 0: max_cases = int(sys.maxsize)
        logger.info(f"Loading {split} dataset...")
        self.tasks: List[Dict[str, Any]] = []
        ds = load_dataset("deepmind/code_contests", split=split, streaming=False, trust_remote_code=False)
        ds = ds.filter(lambda row: _is_valid(row))
        count = 0
        for row in tqdm(ds, desc="Loading tasks", total=ds.num_rows):
            if count >= max_problems:
                break
            
            name, desc = row['name'], row['description']
            ins, outs = row['private_tests']['input'], row['private_tests']['output']
            
            if 'seconds' in row['time_limit']:
                time_limit = row['time_limit']['seconds']
            elif 'nanos' in row['time_limit']:
                time_limit = row['time_limit']['nanos'] / 1e9
            
            self.tasks.append({
                "name": name,
                "prompt": desc,
                "difficulty": row['cf_rating'],
                "tests": list(zip(ins, outs))[:max_cases],
                "time_limit": time_limit,
            })
            count += 1
        logger.info(f"Loaded {len(self.tasks)} valid tasks")
        

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks

def extract_solve_function(response: str, code_pattern: re.Pattern) -> callable:
    """Extract solve function from model response"""
    code_match = code_pattern.search(response)
    code_str = code_match.group(1).strip() if code_match else response.strip()
    
    # Quick validation before exec
    if not code_str or 'def solve' not in code_str:
        return None
        
    namespace = {}
    try:
        exec(code_str, namespace)
        return namespace.get("solve", None)
    except Exception as e:
        logger.debug(f"Failed to extract solve function: {e}")
        return None

def evaluate_single_problem(problem_data: Tuple[int, Dict[str, Any], str, str],
                            code_pattern: re.Pattern) -> List[Dict[str, Any]]:
    """
    Evaluate a single problem – enforces per-test timeouts via `signal.alarm`
    (no ThreadPoolExecutor), avoiding leftover threads at exit.
    """
    import signal
    import logging

    logger = logging.getLogger("eval")

    idx, task, response, filepath = problem_data
    tests        = task["tests"]         # note: changed from 'tests_public' to 'tests'
    description  = task.get("prompt", "")
    problem_name = task.get("name", f"task_{idx}")
    timeout: float = task["time_limit"]  # may be float seconds

    # For human-readable timeout in logs:
    t_int = int(timeout)
    h, m, s = t_int // 3600, (t_int % 3600) // 60, t_int % 60
    print(
        f"Evaluating {problem_name} - {description[:50]}..."
        f" (Timeout: {h}h {m}m {s}s, N tests: {len(tests)})"
    )

    # signal handler raises TimeoutError in this process if alarm triggers
    def _timeout_handler(signum, frame):
        raise TimeoutError()

    solve_fn = extract_solve_function(response, code_pattern)
    if solve_fn is None:
        # If extraction failed, return a failed record for each test
        return [
            {
                "problem_name": problem_name,
                "description": description,
                "test_case": inp,
                "expected_output": expected,
                "generated_output": None,
                "execution_error": "Failed to extract solve function",
                "difficulty": task.get("difficulty", "unknown"),
                "passed": False,
            }
            for inp, expected in tests
        ]

    results: List[Dict[str, Any]] = []
    for idx_test, (inp, expected) in enumerate(tests):
        exec_error, passed, gen_out = "", False, None

        # Register the timeout handler and arm the alarm
        original_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout))

        try:
            gen_out = solve_fn(inp)
            # Cancel any pending alarm
            signal.alarm(0)
            if str(gen_out).strip() == str(expected).strip():
                passed = True
        except TimeoutError:
            exec_error = f"Timed-out after {timeout:.2f}s"
            logger.warning(f"[{problem_name}][{idx_test}] {exec_error}")
        except Exception as e:
            exec_error = str(e)
        finally:
            # Ensure alarm is off and restore previous handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

        results.append(
            {
                "problem_name": problem_name,
                "description": description,
                "test_case": inp,
                "expected_output": expected,
                "generated_output": gen_out,
                "execution_error": exec_error,
                "difficulty": task.get("difficulty", "unknown"),
                "passed": passed,
            }
        )

    return results

class CodeTester:
    def __init__(self, dataset, batch_size: int = 32, max_workers: int = 8, log_file: Optional[str] = None):
        self.tasks = dataset.get_all_tasks()
        self.total = len(self.tasks)
        self.batch_size = batch_size
        self.max_workers = max_workers  # Number of parallel evaluation workers
        self.sample_dir = Path("generated_samples")
        if log_file:
            self.log_file = Path(log_file)
        else:
            self.log_file = "execution_log.csv"
        # Pre-compile regex for better performance
        self.code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    
    def extract_solve_function(self, response: str) -> callable:
        """Extract solve function from model response - instance method"""
        return extract_solve_function(response, self.code_pattern)
    
    def ask_qwen_batch_optimized(
        self, tokenizer, model, prompts: List[str],
        return_hidden: bool = False, return_logprobs: bool = False, num_return_sequences: int = 1
    ) -> Union[List[str], Tuple]:
        """Highly optimized batch generation with optional hidden state and log prob return"""
        responses = []
        all_hidden_states = []
        all_generated_tokens = []
        all_log_probs = []

        # Pre-build all chat templates (vectorized approach)
        all_texts = []
        user_msg_template = (
            "Write a Python function `solve(input_str: str) -> str` that solves the following problem. "
            "Your code should read from the input string and return the correct output string.\n\n"
            "### Problem Description\n{}\n"
            "Only provide the code (no explanations), wrapped in ```python``` markers."
        )

        for prompt in prompts:
            user_msg = user_msg_template.format(prompt)
            msg_sequence = [
                {"role": "system", "content": "You are Qwen, a helpful coding assistant."},
                {"role": "user", "content": user_msg}
            ]
            text = tokenizer.apply_chat_template(msg_sequence, tokenize=False, add_generation_prompt=True)
            all_texts.append(text)

        # Process in optimized batches
        for i in trange(0, len(all_texts), self.batch_size, desc="Generating", leave=True):
            chunk_texts = all_texts[i:i+self.batch_size]
            # Efficient tokenization with proper padding
            inputs = tokenizer(
                chunk_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Limit input length for faster processing
            ).to(model.device)

            # --- Unified block for hidden/logprobs/generation ---
            with torch.no_grad():
                # Compute hidden states if requested
                if return_hidden:
                    forward_outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                    hidden_layer = forward_outputs.hidden_states[-1]
                    seq_lens = inputs.attention_mask.sum(dim=1) - 1
                    batch_hidden = hidden_layer[torch.arange(hidden_layer.size(0)), seq_lens, :]
                    all_hidden_states.append(batch_hidden.cpu())
                # Generate all candidate sequences
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    min_new_tokens=10,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=return_logprobs,
                )
            # Slice off the prompt tokens
            gen_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
            decoded_all = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_generated_tokens.append(gen_ids.cpu())
            # Compute log probabilities if requested
            if return_logprobs and outputs.scores:
                batch_log_probs = []
                for t_idx, token_seq in enumerate(gen_ids):
                    seq_log_probs = []
                    for tok_pos, tok_id in enumerate(token_seq):
                        if tok_pos < len(outputs.scores):
                            logits = outputs.scores[tok_pos][t_idx]
                            logp = torch.nn.functional.log_softmax(logits, dim=-1)[tok_id]
                            seq_log_probs.append(logp.cpu())
                    if seq_log_probs:
                        batch_log_probs.append(torch.stack(seq_log_probs).sum())
                    else:
                        batch_log_probs.append(torch.tensor(0.0))
                all_log_probs.extend(batch_log_probs)
            elif return_logprobs:
                all_log_probs.extend([torch.tensor(0.0) for _ in range(gen_ids.size(0))])
            # Append all decoded responses (no selection)
            responses.extend(decoded_all)

            # Clear cache to prevent memory buildup
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        # Prepare return values
        result = [responses]
        if return_hidden:
            combined_hidden = torch.cat(all_hidden_states, dim=0) if all_hidden_states else torch.empty(0)
            result.append(combined_hidden)
        if return_logprobs:
            combined_tokens = torch.cat(all_generated_tokens, dim=0) if all_generated_tokens else torch.empty(0)
            combined_log_probs = torch.stack(all_log_probs) if all_log_probs else torch.empty(0)
            result.extend([combined_tokens, combined_log_probs])
        return tuple(result) if len(result) > 1 else result[0]
    
    def save_generated_code(self, task_name: str, response: str, idx: int):
        """Save generated code to sample directory"""
        self.sample_dir.mkdir(exist_ok=True)
        
        # Clean task name for filename (remove invalid characters)
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', task_name.strip())
        if not clean_name:
            clean_name = f"task_{idx}"
        
        filename = f"{clean_name}.py"
        filepath = self.sample_dir / filename
        
        # Handle duplicate filenames
        counter = 1
        while filepath.exists():
            filename = f"{clean_name}_{counter}.py"
            filepath = self.sample_dir / filename
            counter += 1
        ## remove ```python``` markers from response
        response = re.sub(r'```python', '', response)
        response = re.sub(r'```', '', response).strip()
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Task: {task_name}\n")
                f.write(f"# Generated solution\n\n")
                f.write(response)
            logger.debug(f"Saved code to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save code for task {task_name}: {e}")
        return filepath

    def run(
        self,
        tokenizer,
        model,
        return_hidden: bool = False,
        return_logprobs: bool = False,
        log: bool = True,
        num_return_seqs: int = 1,
    ) -> Union[float, Tuple]:
        """
        Optimized main evaluation loop with signal-based per-test timeouts and
        non-blocking ProcessPoolExecutor shutdown (using spawn).
        """
        import multiprocessing

        # 1) Extract prompts
        prompts = [task["prompt"] for task in self.tasks]

        # 2) Generate solutions in batches
        print("Generating solutions...")
        generation_result = self.ask_qwen_batch_optimized(
            tokenizer,
            model,
            prompts,
            return_hidden=return_hidden,
            return_logprobs=return_logprobs,
            num_return_sequences=num_return_seqs,
        )

        # 3) Unpack generation result
        if return_hidden and return_logprobs:
            responses, hidden_states, generated_tokens, log_probs = generation_result
            print(
                f"Generated {len(responses)} responses "
                f"with hidden states shape: {hidden_states.shape}"
            )
            print(
                f"Generated tokens shape: {generated_tokens.shape}, "
                f"Log probs shape: {log_probs.shape}"
            )
        elif return_hidden:
            responses, hidden_states = generation_result
            print(
                f"Generated {len(responses)} responses "
                f"with hidden states shape: {hidden_states.shape}"
            )
            generated_tokens, log_probs = None, None
        elif return_logprobs:
            responses, generated_tokens, log_probs = generation_result
            print(f"Generated {len(responses)} responses with tokens and log probs")
            hidden_states = None
        else:
            responses = generation_result
            hidden_states, generated_tokens, log_probs = None, None, None

        # 4) Save each generated code snippet to disk
        print("Saving generated solutions...")
        filepaths: List[str] = []
        for idx, (task, response) in enumerate(zip(self.tasks, responses)):
            task_name = task.get("name", f"task_{idx}")
            filepath = self.save_generated_code(task_name, response, idx)
            filepaths.append(str(filepath))

        # 5) Build argument tuples for parallel evaluation
        problem_data = [
            (idx, task, response, filepaths[idx])
            for idx, (task, response) in enumerate(zip(self.tasks, responses))
        ]

        all_results: List[Dict[str, Any]] = []

        # 6) Use a spawn-based ProcessPoolExecutor to avoid fork/threads deadlocks
        ctx = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=ctx
        )
        try:
            futures = [
                executor.submit(evaluate_single_problem, pd, self.code_pattern)
                for pd in problem_data
            ]

            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Evaluating problems",
                leave=True,
            ):
                try:
                    all_results.extend(fut.result())
                except Exception as e:
                    logger.warning(f"Problem evaluation failed: {e}")
        finally:
            # Non-blocking shutdown: if any worker process is still alive, this
            # lets the main script proceed without hanging.
            executor.shutdown(wait=False, cancel_futures=True)

        print("Evaluation completed. Processing results...")

        # 7) Optionally write CSV log
        if log:
            print("Saving execution results to CSV file...")
            import csv

            with open(self.log_file, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "problem_name",
                    "description",
                    "test_case",
                    "difficulty",
                    "expected_output",
                    "generated_output",
                    "execution_error",
                    "passed",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in all_results:
                    writer.writerow(record)

        # 8) Print summary stats
        pass_counts = sum(1 for r in all_results if r["passed"])
        total_tests = len(all_results)
        avg_rate = (pass_counts / total_tests * 100) if total_tests > 0 else 0.0
        print(f"Average pass rate across tasks: {avg_rate:.2f}%")

        err_count = sum(1 for r in all_results if r["execution_error"])
        err_rate = (err_count / total_tests * 100) if total_tests > 0 else 0.0
        print(f"Execution error rate: {err_rate:.2f}% ({err_count}/{total_tests})")

        # 9) Return based on requested flags
        if return_hidden and return_logprobs:
            return avg_rate, responses, hidden_states, generated_tokens, log_probs
        elif return_hidden:
            return avg_rate, responses, hidden_states
        elif return_logprobs:
            return avg_rate, responses, generated_tokens, log_probs
        else:
            return avg_rate

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel tokenization to avoid issues
    # Enable optimizations
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    
    dataset = CodeContestDataset(split="test")
    tokenizer, model = load_qwen_model(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    tester = CodeTester(dataset=dataset, batch_size=200, max_workers=cpu_count() // 2)
    
    tester.run(tokenizer, model, log=False)

if __name__ == "__main__":
    main()