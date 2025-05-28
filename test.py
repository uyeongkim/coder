import io
import sys
from tqdm import tqdm
import re
from typing import Any, Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_tasks(split: str = "train", max_problems: int = 500):
    logger.info(f"Loading {split} dataset...")
    tasks = []
    try:
        ds = load_dataset("deepmind/code_contests", split=split, streaming=True, trust_remote_code=False)
        count = 0
        for row in ds:
            if count >= max_problems:
                break
            try:
                ins = row.get("public_tests", {}).get("input", [])
                outs = row.get("public_tests", {}).get("output", [])
                gt_solution = row.get("solutions", {})
                if 1 in gt_solution.get("language", []):
                    idx = gt_solution["language"].index(1)
                    solution = gt_solution["solution"][idx]
                    if 20 < len(solution.strip()) < 3000:
                        tasks.append({
                            "name": row.get("name", ""),
                            "prompt": row.get("description", "")[:1000],
                            "tests_public": list(zip(ins, outs))[:3]
                        })
                        count += 1
            except Exception as e:
                logger.warning(f"Skipping problematic row: {e}")
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
    logger.info(f"Loaded {len(tasks)} valid tasks")
    return tasks

def load_qwen_model(model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return tokenizer, model

def ask_qwen_to_solve(tokenizer, model, prompt: str) -> str:
    user_msg = (
        "Write a Python function `solve(input_str: str) -> str` that solves the following problem. "
        "Your code should read from the input string and return the correct output string.\n\n"
        f"### Problem Description\n{prompt}\n"
        "Only provide the code (no explanations), wrapped in ```python``` markers."
    )
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful coding assistant."},
        {"role": "user", "content": user_msg}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=1024)
    gen_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

def extract_solve_function(response: str):
    code_match = re.search(r'```python(.*?)```', response, re.DOTALL)
    code_str = code_match.group(1) if code_match else response
    namespace = {}
    try:
        exec(code_str, namespace)
    except:
        return None
    return namespace.get("solve", None)

def run_tests(solve_func, tests):
    if not solve_func:
        print("No valid solve() function defined.")
        return 0, len(tests), 0.0
    passed = 0
    for idx, (inp, expected) in tqdm(enumerate(tests, 1), desc="Running tests", total=len(tests), leave=False):
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(solve_func, inp)
                result = future.result(timeout=10)
        except Exception as e:
            print(f"Test {idx} raised an exception: {e}")
            continue
        res = str(result).strip()
        exp = str(expected).strip()
        ok = (res == exp)
        status = '✓' if ok else '✗'
        print(f"Test {idx}: {status}")
        print(f"  Input:    {inp!r}\n  Expected: {exp!r}\n  Got:      {res!r}")
        if ok:
            passed += 1
    total = len(tests)
    rate = passed / total * 100 if total > 0 else 0.0
    print(f"\nPassed {passed}/{total} tests ({rate:.2f}% pass rate)")
    return passed, total, rate

class CodeContestDataset:
    def __init__(self, split: str = "train", max_problems: int = 50):
        logger.info(f"Loading {split} dataset...")
        self.tasks: List[Dict[str, Any]] = []
        try:
            ds = load_dataset("deepmind/code_contests", split=split, streaming=True, trust_remote_code=False)
            count = 0
            for row in ds:
                if count >= max_problems:
                    break
                try:
                    ins = row.get("public_tests", {}).get("input", [])
                    outs = row.get("public_tests", {}).get("output", [])
                    gt_solution = row.get("solutions", {})
                    if 1 in gt_solution.get("language", []):
                        idx = gt_solution["language"].index(1)
                        solution = gt_solution["solution"][idx]
                        if 20 < len(solution.strip()) < 3000:
                            self.tasks.append({
                                "name": row.get("name", ""),
                                "prompt": row.get("description", "")[:1000],
                                "tests_public": list(zip(ins, outs))
                            })
                            count += 1
                except Exception as e:
                    logger.warning(f"Skipping problematic row: {e}")
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
        logger.info(f"Loaded {len(self.tasks)} valid tasks")

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks

def main():
    tasks = load_tasks(split="train", max_problems=50)
    if not tasks:
        print("No tasks loaded.")
        return

    tokenizer, model = load_qwen_model()
    pass_rates = []
    for task in tasks:
        print(f"Task: {task['name']}")
        print(f"Prompt: {task['prompt'][:100]}...")
        response = ask_qwen_to_solve(tokenizer, model, task['prompt'])

        solve_fn = extract_solve_function(response)
        _, _, rate = run_tests(solve_fn, task["tests_public"])
        pass_rates.append(rate)
        
    print(f"Average pass rate across tasks: {sum(pass_rates) / len(pass_rates):.2f}%")

if __name__ == "__main__":
    main()