import os, sys, random, logging, re, csv, signal, concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
import math
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from marcog_config import (
    GlobalSHPPOConfig, ACTION_TO_IDX, IDX_TO_ACTION, ACTION_METADATA,
    PLANNER_ACTION_INDICES, CODER_ACTION_INDICES, DEBUGGER_ACTION_INDICES,
)

logger = logging.getLogger(__name__)

def load_qwen_model(model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct", model_dtype: torch.dtype = torch.bfloat16):
    """Loads a Qwen model and tokenizer with 4-bit quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=model_dtype,
        bnb_4bit_use_double_quant=True, llm_int8_enable_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    logger.info(f"Model {model_name} loaded successfully for standalone testing with dtype {model.dtype}.")
    return tokenizer, model

class CodeContestDataset:
    def __init__(self, split: str = "train", max_problems: int = -1, max_cases: int = -1):
        def _is_valid(row):
            if row is None: return False
            if row.get('input_file') or row.get('output_file'): return False
            if row.get('description') is None or not (10 < len(row['description']) < 1500): return False
            if row.get('name') is None or row.get('private_tests') is None or len(row['private_tests']['input']) == 0: return False
            if row.get('solutions') is None or 1 not in row['solutions']['language'] or not any(len(sol.strip()) < 500 for sol in row['solutions']['solution']): return False
            if row.get('time_limit') is None or not (row['time_limit'].get('seconds') or row['time_limit'].get('nanos')): return False
            if row.get('cf_rating') is None: return False
            return True

        if max_problems <= 0: max_problems = int(sys.maxsize)
        if max_cases <= 0: max_cases = int(sys.maxsize)
        logger.info(f"Loading {split} dataset...")
        self.tasks: List[Dict[str, Any]] = []
        try:
            ds = load_dataset("deepmind/code_contests", split=split, streaming=False, trust_remote_code=False)
        except Exception as e:
            logger.error(f"Failed to load dataset 'deepmind/code_contests': {e}")
            raise

        ds = ds.filter(lambda row: _is_valid(row))
        count = 0
        for row in tqdm(ds, desc="Loading tasks", total=ds.num_rows if hasattr(ds, "num_rows") else None):
            if count >= max_problems: break
            name, desc = row['name'], row['description']
            ins, outs = row['private_tests']['input'], row['private_tests']['output']
            time_limit_val = row['time_limit'].get('seconds')
            if time_limit_val is None:
                time_limit_val = row['time_limit'].get('nanos', 0) / 1e9

            self.tasks.append({
                "name": name, "prompt": desc, "difficulty": row['cf_rating'],
                "tests": list(zip(ins, outs))[:max_cases], "time_limit": time_limit_val,
            })
            count += 1
        logger.info(f"Loaded {len(self.tasks)} valid tasks from {split} split.")

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks

def extract_solve_function(response: str, code_pattern: re.Pattern) -> Optional[callable]:
    code_match = code_pattern.search(response)
    code_str = code_match.group(1).strip() if code_match else response.strip()
    if not code_str or 'def solve' not in code_str: return None
    namespace = {}
    try:
        exec(code_str, namespace)
        return namespace.get("solve", None)
    except Exception as e:
        logger.debug(f"Failed to extract solve function: {e}")
        return None

def evaluate_single_problem(problem_data: Tuple[int, Dict[str, Any], str, str],
                            code_pattern: re.Pattern) -> List[Dict[str, Any]]:
    idx, task, response, filepath = problem_data
    tests = task["tests"]
    description = task.get("prompt", "")
    problem_name = task.get("name", f"task_{idx}")
    timeout: float = float(task.get("time_limit", 10.0))

    def _timeout_handler(signum, frame):
        raise TimeoutError()

    solve_fn = extract_solve_function(response, code_pattern)
    if solve_fn is None:
        return [
            {"problem_name": problem_name, "description": description, "test_case": inp,
             "expected_output": expected, "generated_output": None,
             "execution_error": "Failed to extract solve function",
             "difficulty": task.get("difficulty", "unknown"), "passed": False}
            for inp, expected in tests
        ]
    results: List[Dict[str, Any]] = []
    for idx_test, (inp, expected) in enumerate(tests):
        exec_error, passed, gen_out_str = "", False, None
        original_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(math.ceil(timeout)))
        try:
            gen_out = solve_fn(inp)
            gen_out_str = str(gen_out).strip()
            signal.alarm(0)
            if gen_out_str == str(expected).strip(): passed = True
        except TimeoutError:
            exec_error = f"Timed-out after {timeout:.2f}s"
        except Exception as e:
            exec_error = str(e)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
        results.append({
            "problem_name": problem_name, "description": description, "test_case": inp,
            "expected_output": expected, "generated_output": gen_out_str,
            "execution_error": exec_error, "difficulty": task.get("difficulty", "unknown"),
            "passed": passed,
        })
    return results

def format_test_results_for_llm(eval_results: List[Dict[str, Any]]) -> str:
    if not eval_results: return "No test results available or no code to test."
    formatted_string = "### Detailed Test Results:\n"
    for i, res in enumerate(eval_results):
        formatted_string += f"--- Test Case {i+1} ---\n"
        formatted_string += f"Input:\n```\n{res['test_case'].strip()}\n```\n"
        formatted_string += f"Expected Output:\n```\n{res['expected_output'].strip()}\n```\n"
        generated_output_str = str(res['generated_output']).strip() if res['generated_output'] is not None else "None"
        formatted_string += f"Generated Output:\n```\n{generated_output_str}\n```\n"
        formatted_string += f"Passed: {res['passed']}\n"
        if res['execution_error']:
            formatted_string += f"Execution Error: {res['execution_error'].strip()}\n"
        formatted_string += "\n"
    return formatted_string

class CodeTester:
    def __init__(self, dataset, global_config: GlobalSHPPOConfig, tokenizer: AutoTokenizer, llm_model: AutoModelForCausalLM, log_file: Optional[str] = None):
        self.tasks = dataset.get_all_tasks()
        self.total = len(self.tasks)
        self.global_config = global_config
        self.model_dtype = global_config.model_dtype if global_config.model_dtype is not None else torch.bfloat16
        self.sample_dir = Path("generated_samples")
        self.sample_dir.mkdir(exist_ok=True)
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.log_file = Path(log_file) if log_file else Path(f"execution_log_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv")
        self.code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
        self.num_planners = global_config.total_planner_agents
        self.num_coders = global_config.total_coder_agents
        self.num_debuggers = global_config.total_debugger_agents

    def extract_solve_function(self, response: str) -> Optional[callable]:
        return extract_solve_function(response, self.code_pattern)

    def _format_prompt(self, action_name: str, problem_desc: str, context: Optional[str] = None) -> str:
        system_msg = "You are Qwen, a helpful coding assistant."
        user_msg = ""
        if action_name == "plan-subgoal":
            user_msg = (f"### Problem Description\n{problem_desc}\n\n"
                        "You are a Planner. Break down this problem into clear, actionable sub-goals or a step-by-step plan. "
                        "Output ONLY the plan as a concise list of steps. ")
            if context: user_msg += f"\n### Previous Plan / Context\n{context}\nRefine or elaborate on the previous plan based on the problem and the context."
        elif action_name == "generate-code":
            user_msg = (f"### Problem Description\n{problem_desc}\n\n"
                        "You are a Code Generator. Write a Python function `solve(input_str: str) -> str` that solves the problem. "
                        "Your code should read from the input string and return the correct output string.\n"
                        "Output ONLY the code (no explanations), wrapped in ```python``` markers.")
            if context: user_msg += f"\n### Plan / Previous Code / Test Results\n{context}\nImprove and Optimize the previous code based on the context. "
        elif action_name == "fix-code":
            user_msg = (f"### Problem Description\n{problem_desc}\n\n"
                        "You are a Debugger. Analyze the provided code and its execution results. "
                        "Identify and fix any bugs. Output ONLY the fixed code (no explanations), wrapped in ```python``` markers.")
            if context: user_msg += f"\n### Code and Detailed Test Results\n{context}\nFix the code based on the test results. "
        elif action_name == "noop":
            user_msg = "This is a no-operation turn. Do not generate any code or plan. Output nothing."
        else: raise ValueError(f"Unknown action name for prompt formatting: {action_name}")
        msg_sequence = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        return self.tokenizer.apply_chat_template(msg_sequence, tokenize=False, add_generation_prompt=True)

    def _get_llm_output(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, formatted_prompt: str, num_return_sequences: int = 1) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = tokenizer(
            [formatted_prompt], return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)

        llm_dtype = model.dtype

        with torch.no_grad():
            forward_outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_layer = forward_outputs.hidden_states[-1]
            seq_lens = inputs.attention_mask.sum(dim=1) - 1

            llm_hidden_state_on_device = hidden_layer[torch.arange(hidden_layer.size(0)), seq_lens, :]
            llm_hidden_state_cpu = llm_hidden_state_on_device.cpu()

            outputs = model.generate(
                **inputs, max_new_tokens=512, min_new_tokens=10, do_sample=True,
                num_return_sequences=num_return_sequences, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True,
                return_dict_in_generate=True, output_scores=True,
            )
        gen_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        decoded_responses = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        batch_log_probs = []
        for i_seq in range(gen_ids.size(0)):
            seq_log_probs = []
            for i_token, tok_id in enumerate(gen_ids[i_seq]):
                if i_token < len(outputs.scores):
                    logits_at_pos = outputs.scores[i_token][i_seq]
                    logp = F.log_softmax(logits_at_pos.to(torch.float32), dim=-1)[tok_id]
                    seq_log_probs.append(logp.cpu())
            batch_log_probs.append(torch.stack(seq_log_probs).sum() if seq_log_probs else torch.tensor(0.0, dtype=torch.float32))

        llm_log_probs_cpu = torch.stack(batch_log_probs).cpu() if batch_log_probs else torch.empty(0, dtype=torch.float32)
        gen_ids_cpu = gen_ids.cpu()

        if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
        return decoded_responses, llm_hidden_state_cpu.to(llm_dtype), llm_log_probs_cpu, gen_ids_cpu


    def _get_agent_action_and_latent(
        self, actor_net: torch.nn.Module, latent_net: torch.nn.Module,
        llm_hidden_state_cpu: torch.Tensor,
        agent_h_prev_on_device: torch.Tensor,
        problem_difficulty: float, time_limit: float,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        target_device = agent_h_prev_on_device.device
        target_dtype = agent_h_prev_on_device.dtype

        llm_hidden_state_device = llm_hidden_state_cpu.to(target_device, dtype=target_dtype)

        scalar_features_list = [problem_difficulty, time_limit]
        scalar_features = torch.tensor(scalar_features_list, dtype=target_dtype, device=target_device).unsqueeze(0)

        obs_emb = torch.cat([llm_hidden_state_device, scalar_features], dim=-1)

        z_all_templates, mu, sigma = latent_net(obs_emb, agent_h_prev_on_device)
        action_logits, h_next, action_probs = actor_net(obs_emb, agent_h_prev_on_device, z_all_templates)

        dist = torch.distributions.Categorical(logits=action_logits.to(torch.float32))
        chosen_action_flat_idx = dist.sample()
        chosen_action_log_prob = dist.log_prob(chosen_action_flat_idx)

        return chosen_action_flat_idx.item(), z_all_templates, mu, sigma, h_next, chosen_action_log_prob


    def run_pipeline_for_problem(
        self, problem_id_in_batch: int, task: Dict[str, Any],
        tokenizer: AutoTokenizer, llm_model: AutoModelForCausalLM,
        role_networks: Dict[str, Any], critic_net: torch.nn.Module,
        inference_net: torch.nn.Module, initial_h_critic_on_device: torch.Tensor,
    ) -> Dict[str, Any]:
        logger.info(f"RUN_PIPELINE: Entered for problem_id_in_batch: {problem_id_in_batch}, problem_name: {task.get('name', 'N/A')}")
        problem_desc = task["prompt"]
        problem_name = task["name"]
        problem_difficulty = float(task["difficulty"])
        problem_time_limit = float(task["time_limit"])

        model_dtype = self.global_config.model_dtype if self.global_config.model_dtype is not None else llm_model.dtype
        float32_dtype = torch.float32

        problem_ppo_data = {
            "problem_name": problem_name,
            "problem_difficulty": torch.tensor(problem_difficulty, dtype=float32_dtype),
            "problem_time_limit": torch.tensor(problem_time_limit, dtype=float32_dtype),
            "final_reward": None, "initial_value_prediction": None,
            "initial_global_state_embedding": None, "initial_critic_hidden_state": initial_h_critic_on_device.cpu(),
            "all_mus_flat_for_inference": [], "all_sigmas_flat_for_inference": [],
            "all_z_for_diversity": [], "all_mu_for_diversity": [], "all_sigma_for_diversity": [],
            "actor_inputs_by_role": { role: {"obs_embs": [], "h_prevs": [], "z_all_templates": [], "old_log_probs": [], "chosen_actions": [], "advantages": []}
                                     for role in ["planner", "coder", "debugger"] if getattr(self.global_config, f"total_{role}_agents", 0) > 0},
            "llm_responses_per_turn": [], "llm_generated_tokens_per_turn": [],
            "llm_log_probs_per_turn": [], "turn_agent_role_names": [], "turn_action_templates": [],
        }

        # --- Initial Global State (o_g) 구성 ---
        logger.info(f"RUN_PIPELINE: [{problem_name}] Starting initial global state embedding generation.")
        problem_name_text = f"Problem Name: {problem_name}"
        msg_sequence_name = [{"role": "user", "content": problem_name_text}]
        formatted_problem_name_prompt = tokenizer.apply_chat_template(msg_sequence_name, tokenize=False, add_generation_prompt=True)
        _, llm_h_state_problem_name_cpu, _, _ = self._get_llm_output(tokenizer, llm_model, formatted_problem_name_prompt)

        initial_problem_desc_prompt_text = self._format_prompt("plan-subgoal", problem_desc)
        _, llm_h_state_problem_desc_cpu, _, _ = self._get_llm_output(tokenizer, llm_model, initial_problem_desc_prompt_text)
        logger.info(f"RUN_PIPELINE: [{problem_name}] Initial global state embedding generation complete.")

        scalar_features_global = torch.tensor([problem_difficulty, problem_time_limit], dtype=model_dtype, device=llm_model.device).unsqueeze(0)

        current_global_state_emb_device = torch.cat([
            llm_h_state_problem_name_cpu.to(llm_model.device, dtype=model_dtype),
            llm_h_state_problem_desc_cpu.to(llm_model.device, dtype=model_dtype),
            scalar_features_global
        ], dim=-1)
        problem_ppo_data["initial_global_state_embedding"] = current_global_state_emb_device.cpu()

        current_value_pred_device, _ = critic_net(current_global_state_emb_device, initial_h_critic_on_device)
        problem_ppo_data["initial_value_prediction"] = current_value_pred_device.cpu()

        agent_h_prevs: Dict[str, List[torch.Tensor]] = {
            role: [torch.zeros(1, self.global_config.actor_rnn_hidden_dim, device=llm_model.device, dtype=model_dtype)
                   for _ in range(getattr(self.global_config, f"total_{role}_agents", 0))]
            for role in ["planner", "coder", "debugger"] if getattr(self.global_config, f"total_{role}_agents", 0) > 0
        }
        current_plan, current_code, final_code_for_eval = "", "", ""
        latest_eval_results_for_context: List[Dict[str, Any]] = []

        pipeline_sequence_def = []
        if hasattr(self, 'num_planners') and self.num_planners > 0: pipeline_sequence_def.extend([("planner", i) for i in range(self.num_planners)])
        if hasattr(self, 'num_coders') and self.num_coders > 0: pipeline_sequence_def.extend([("coder", i) for i in range(self.num_coders)])
        if hasattr(self, 'num_debuggers') and self.num_debuggers > 0: pipeline_sequence_def.extend([("debugger", i) for i in range(self.num_debuggers)])

        temp_all_agents_mu_flat_list, temp_all_agents_sigma_flat_list = [], []

        logger.info(f"RUN_PIPELINE: [{problem_name}] Starting agent pipeline sequence (total turns: {len(pipeline_sequence_def)}).")
        # === Main pipeline loop ===
        for turn_idx_main_loop, (role_name_main, agent_idx_in_role_main) in enumerate(pipeline_sequence_def):
            logger.info(f"RUN_PIPELINE: [{problem_name}] Beginning Turn {turn_idx_main_loop + 1} - Role: {role_name_main}, Agent_Idx: {agent_idx_in_role_main}")

            if role_name_main not in role_networks or not role_networks[role_name_main]:
                logger.warning(f"RUN_PIPELINE: [{problem_name}] Networks for role {role_name_main} not found, skipping turn {turn_idx_main_loop + 1}")
                problem_ppo_data["turn_agent_role_names"].append(role_name_main)
                problem_ppo_data["turn_action_templates"].append("SKIPPED_NO_NET")
                problem_ppo_data["llm_responses_per_turn"].append("SKIPPED_NO_NET")
                continue
            actor_net = role_networks[role_name_main]["actor_net"]
            latent_net = role_networks[role_name_main]["latent_net"]

            # 1. Construct Context
            logger.info(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - Constructing context.")
            context_for_llm_prompt: Optional[str] = None
            if role_name_main == "planner":
                context_for_llm_prompt = current_plan if agent_idx_in_role_main > 0 else None
            elif role_name_main == "coder":
                context_for_llm_prompt = f"Plan: {current_plan if current_plan else 'No plan yet.'}"
                if agent_idx_in_role_main > 0 and current_code:
                    context_for_llm_prompt += f"\nPrevious Code:\n```python\n{current_code.strip()}\n```\n"
                    if latest_eval_results_for_context:
                        context_for_llm_prompt += format_test_results_for_llm(latest_eval_results_for_context)
            elif role_name_main == "debugger":
                temp_code_for_debug_eval = final_code_for_eval if final_code_for_eval else current_code
                if temp_code_for_debug_eval:
                    logger.info(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - Debugger evaluating code for context.")
                    temp_filepath_debug_eval = self.sample_dir / f"temp_debug_eval_{problem_name.replace('/', '_')}_{role_name_main}_{agent_idx_in_role_main}.py"
                    clean_code_debug_eval = re.sub(r'```python', '', temp_code_for_debug_eval, flags=re.IGNORECASE); clean_code_debug_eval = re.sub(r'```', '', clean_code_debug_eval).strip()
                    try:
                        with open(temp_filepath_debug_eval, 'w', encoding='utf-8') as f: f.write(clean_code_debug_eval)
                        problem_data_for_eval_debug = (problem_id_in_batch, task, temp_code_for_debug_eval, str(temp_filepath_debug_eval))
                        current_debug_eval_results = evaluate_single_problem(problem_data_for_eval_debug, self.code_pattern)
                        context_for_llm_prompt = f"Code:\n```python\n{temp_code_for_debug_eval.strip()}\n```\n{format_test_results_for_llm(current_debug_eval_results)}"
                    except Exception as e_debug_eval:
                        logger.error(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - Error during debugger's temp eval: {e_debug_eval}", exc_info=True)
                        context_for_llm_prompt = f"Code:\n```python\n{temp_code_for_debug_eval.strip()}\n```\n\nError during its evaluation: {str(e_debug_eval)}"
                    finally:
                        if temp_filepath_debug_eval.exists(): os.remove(temp_filepath_debug_eval)
                else: context_for_llm_prompt = "Code: No code generated yet."

            # 2. Generate LLM Candidates
            logger.info(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - Context constructed. Generating LLM candidates.")
            all_llm_candidate_data_for_this_turn = []
            valid_action_indices_for_role_set = set()
            if role_name_main == "planner": valid_action_indices_for_role_set = PLANNER_ACTION_INDICES
            elif role_name_main == "coder": valid_action_indices_for_role_set = CODER_ACTION_INDICES
            elif role_name_main == "debugger": valid_action_indices_for_role_set = DEBUGGER_ACTION_INDICES
            role_ordered_valid_global_indices = sorted(list(valid_action_indices_for_role_set))

            for action_template_global_idx in role_ordered_valid_global_indices:
                action_name_candidate = IDX_TO_ACTION[action_template_global_idx]
                metadata = ACTION_METADATA[action_name_candidate]
                logger.info(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - Generating for action: {action_name_candidate}")
                if metadata["uses_llm"]:
                    formatted_prompt_for_template = self._format_prompt(action_name_candidate, problem_desc, context_for_llm_prompt)
                    responses, llm_h_state_batch_cpu, log_probs_batch_cpu, gen_tokens_batch_cpu = self._get_llm_output(
                        tokenizer, llm_model, formatted_prompt_for_template, num_return_sequences=metadata["sample_count"]
                    )
                    for i in range(metadata["sample_count"]):
                        all_llm_candidate_data_for_this_turn.append({
                            "action_name": action_name_candidate, "response": responses[i],
                            "llm_hidden_state": llm_h_state_batch_cpu[i, :].clone() if llm_h_state_batch_cpu.ndim == 2 and llm_h_state_batch_cpu.shape[0] == metadata["sample_count"] else llm_h_state_batch_cpu.squeeze(0).clone(),
                            "llm_log_prob": log_probs_batch_cpu[i].clone(),
                            "llm_generated_tokens": gen_tokens_batch_cpu[i].clone()
                        })
                elif metadata["is_fixed_response"]:
                    for _ in range(metadata["sample_count"]):
                        all_llm_candidate_data_for_this_turn.append({
                            "action_name": action_name_candidate, "response": "",
                            "llm_hidden_state": torch.zeros(self.global_config.llm_actual_hidden_size if self.global_config.llm_actual_hidden_size else llm_model.config.hidden_size, device='cpu', dtype=model_dtype),
                            "llm_log_prob": torch.tensor(0.0, device='cpu', dtype=float32_dtype),
                            "llm_generated_tokens": torch.tensor([], dtype=torch.long, device='cpu')
                        })

            if not all_llm_candidate_data_for_this_turn:
                logger.warning(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - No LLM candidates generated. Skipping turn.")
                continue
            logger.info(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - LLM candidates generated ({len(all_llm_candidate_data_for_this_turn)} total).")

            # 3. Representative LLM Hidden State
            representative_llm_hidden_state_cpu = all_llm_candidate_data_for_this_turn[0]["llm_hidden_state"].unsqueeze(0)

            # 4. Agent Action Selection
            logger.info(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - Agent selecting action.")
            chosen_action_flat_idx, z_all_templates, mu_agent, sigma_agent, h_actor_next, chosen_action_log_prob = \
                self._get_agent_action_and_latent(
                    actor_net, latent_net, representative_llm_hidden_state_cpu,
                    agent_h_prevs[role_name_main][agent_idx_in_role_main],
                    problem_difficulty, problem_time_limit
                )
            logger.info(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - chosen_action_flat_idx: {chosen_action_flat_idx}")


            # --- 5. Parse and Extract Response ---
            chosen_action_template_global_idx_parsed, chosen_llm_candidate_idx_parsed, chosen_action_name_parsed = -1, -1, "ERROR_ACTION"
            current_offset, found_action_parse = 0, False
            for template_global_idx_iter in role_ordered_valid_global_indices:
                action_name_iter = IDX_TO_ACTION[template_global_idx_iter]
                metadata_iter = ACTION_METADATA[action_name_iter]
                sample_count_for_template = metadata_iter["sample_count"]
                if chosen_action_flat_idx >= current_offset and chosen_action_flat_idx < current_offset + sample_count_for_template:
                    chosen_action_template_global_idx_parsed = template_global_idx_iter
                    chosen_llm_candidate_idx_parsed = chosen_action_flat_idx - current_offset
                    chosen_action_name_parsed = action_name_iter
                    found_action_parse = True
                    break
                current_offset += sample_count_for_template

            final_agent_response = "ERROR_DEFAULT_RESPONSE"
            final_llm_generated_tokens_cpu = torch.tensor([], dtype=torch.long, device='cpu')
            final_llm_log_prob_cpu = torch.tensor(-1e9, device='cpu', dtype=float32_dtype)

            if not found_action_parse:
                final_agent_response = f"ERROR_GENERATION_PARSE_FAIL_IDX_{chosen_action_flat_idx}"
                logger.error(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - {final_agent_response}")
            else:
                candidate_list_scan_offset, retrieved_data_successfully = 0, False
                for template_global_idx_data_retrieval in role_ordered_valid_global_indices:
                    action_name_data_retrieval = IDX_TO_ACTION[template_global_idx_data_retrieval]
                    metadata_data_retrieval = ACTION_METADATA[action_name_data_retrieval]
                    if action_name_data_retrieval == chosen_action_name_parsed:
                        actual_list_index = candidate_list_scan_offset + chosen_llm_candidate_idx_parsed
                        if actual_list_index < len(all_llm_candidate_data_for_this_turn) and \
                           all_llm_candidate_data_for_this_turn[actual_list_index]["action_name"] == chosen_action_name_parsed:
                            final_agent_response_data_obj = all_llm_candidate_data_for_this_turn[actual_list_index]
                            final_agent_response = final_agent_response_data_obj["response"]
                            final_llm_generated_tokens_cpu = final_agent_response_data_obj["llm_generated_tokens"]
                            final_llm_log_prob_cpu = final_agent_response_data_obj["llm_log_prob"]
                            retrieved_data_successfully = True
                            break
                    candidate_list_scan_offset += metadata_data_retrieval["sample_count"]
                if not retrieved_data_successfully:
                    final_agent_response = f"ERROR_RETRIEVAL_FAIL_ACTION_{chosen_action_name_parsed}_CAND_{chosen_llm_candidate_idx_parsed}"
                    logger.error(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - {final_agent_response}")

            logger.info(f"PROBLEM_TURN_LOG: [{task.get('name', f'Problem_ID_{problem_id_in_batch}')}] - TURN {turn_idx_main_loop + 1}: Role [{role_name_main}], Agent_Idx [{agent_idx_in_role_main}], Chosen Action -> [{chosen_action_name_parsed}]")

            # --- 6. Update Pipeline State ---
            logger.info(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - Updating pipeline state with action: {chosen_action_name_parsed}")
            if chosen_action_name_parsed == "plan-subgoal": current_plan = final_agent_response
            elif chosen_action_name_parsed == "generate-code":
                current_code = final_agent_response; final_code_for_eval = current_code
                if current_code and not current_code.startswith("ERROR_"):
                    temp_filepath_eval_coder = self.sample_dir / f"temp_coder_eval_{problem_name.replace('/', '_')}_{role_name_main}_{agent_idx_in_role_main}.py"
                    clean_code_coder = re.sub(r'```python', '', current_code, flags=re.IGNORECASE); clean_code_coder = re.sub(r'```', '', clean_code_coder).strip()
                    try:
                        with open(temp_filepath_eval_coder, 'w', encoding='utf-8') as f: f.write(clean_code_coder)
                        problem_data_for_eval_coder = (problem_id_in_batch, task, current_code, str(temp_filepath_eval_coder))
                        latest_eval_results_for_context = evaluate_single_problem(problem_data_for_eval_coder, self.code_pattern)
                    except Exception as e_coder_temp_eval:
                        logger.error(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - Error in coder temp eval: {e_coder_temp_eval}", exc_info=True)
                        latest_eval_results_for_context = []
                    finally:
                        if temp_filepath_eval_coder.exists(): os.remove(temp_filepath_eval_coder)
                else: latest_eval_results_for_context = []
            elif chosen_action_name_parsed == "fix-code": final_code_for_eval = final_agent_response

            # --- 7. Store PPO Data ---
            logger.info(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - Storing PPO data.")
            problem_ppo_data["actor_inputs_by_role"][role_name_main]["obs_embs"].append(
                torch.cat([representative_llm_hidden_state_cpu.squeeze(0),
                           torch.tensor([problem_difficulty, problem_time_limit], dtype=model_dtype).cpu()], dim=-1)
            )
            problem_ppo_data["actor_inputs_by_role"][role_name_main]["h_prevs"].append(agent_h_prevs[role_name_main][agent_idx_in_role_main].cpu())
            problem_ppo_data["actor_inputs_by_role"][role_name_main]["z_all_templates"].append(z_all_templates.cpu())
            problem_ppo_data["actor_inputs_by_role"][role_name_main]["old_log_probs"].append(chosen_action_log_prob.cpu())
            problem_ppo_data["actor_inputs_by_role"][role_name_main]["chosen_actions"].append(torch.tensor(chosen_action_flat_idx, dtype=torch.long).cpu())

            problem_ppo_data["llm_responses_per_turn"].append(str(final_agent_response))
            problem_ppo_data["llm_generated_tokens_per_turn"].append(final_llm_generated_tokens_cpu)
            problem_ppo_data["llm_log_probs_per_turn"].append(final_llm_log_prob_cpu)
            problem_ppo_data["turn_agent_role_names"].append(role_name_main)
            problem_ppo_data["turn_action_templates"].append(chosen_action_name_parsed)

            temp_all_agents_mu_flat_list.append(mu_agent.reshape(mu_agent.size(0), -1).cpu())
            temp_all_agents_sigma_flat_list.append(sigma_agent.reshape(sigma_agent.size(0), -1).cpu())
            problem_ppo_data["all_z_for_diversity"].append(z_all_templates.mean(dim=1).cpu())
            problem_ppo_data["all_mu_for_diversity"].append(mu_agent.cpu())
            problem_ppo_data["all_sigma_for_diversity"].append(sigma_agent.cpu())

            # --- 8. Update Agent Hidden State ---
            agent_h_prevs[role_name_main][agent_idx_in_role_main] = h_actor_next
            logger.info(f"RUN_PIPELINE: [{problem_name}] Turn {turn_idx_main_loop + 1} - Completed.")
        # === End Main pipeline loop ===

        logger.info(f"RUN_PIPELINE: [{problem_name}] Agent pipeline sequence finished. Starting final evaluation.")
        # --- Final Evaluation and Reward Calculation ---
        final_evaluation_results: List[Dict[str, Any]] = []
        if final_code_for_eval and not final_code_for_eval.startswith("ERROR_"):
            final_filepath = self.sample_dir / f"final_code_{problem_name.replace('/', '_')}.py"
            clean_code = re.sub(r'```python', '', final_code_for_eval, flags=re.IGNORECASE); clean_code = re.sub(r'```', '', clean_code).strip()
            try:
                with open(final_filepath, 'w', encoding='utf-8') as f: f.write(clean_code)
                final_problem_data_for_eval = (problem_id_in_batch, task, final_code_for_eval, str(final_filepath))
                final_evaluation_results = evaluate_single_problem(final_problem_data_for_eval, self.code_pattern)
            except Exception as e_final_eval:
                logger.error(f"RUN_PIPELINE: [{problem_name}] Error during final code evaluation: {e_final_eval}", exc_info=True)
                if task.get("tests"):
                    final_evaluation_results = [{"passed": False, "execution_error": str(e_final_eval)} for _ in task["tests"]]

        per_test_rewards = []
        for res in final_evaluation_results:
            if res['passed']: per_test_rewards.append(1.0)
            elif isinstance(res.get('execution_error'), str) and res['execution_error'] != "": per_test_rewards.append(-1.0)
            else: per_test_rewards.append(0.0)
        problem_reward = sum(per_test_rewards) / len(per_test_rewards) if per_test_rewards else 0.0
        problem_ppo_data["final_reward"] = torch.tensor(problem_reward, dtype=float32_dtype)

        passed_count_for_this_problem = 0
        total_tests_for_this_problem = 0
        if final_evaluation_results:
            total_tests_for_this_problem = len(final_evaluation_results)
            for res_check in final_evaluation_results:
                if res_check['passed']:
                    passed_count_for_this_problem += 1
        elif task.get("tests"):
            total_tests_for_this_problem = len(task["tests"])

        problem_pass_rate_for_this_problem = (passed_count_for_this_problem / total_tests_for_this_problem * 100) if total_tests_for_this_problem > 0 else 0.0
        logger.info(f"RUN_PIPELINE_PROBLEM_RESULT: Problem [{problem_name}] Pass Rate: {problem_pass_rate_for_this_problem:.2f}% ({passed_count_for_this_problem}/{total_tests_for_this_problem})")

        # --- Data Cleanup and Return ---
        if temp_all_agents_mu_flat_list:
            problem_ppo_data["all_mus_flat_for_inference"] = torch.cat(temp_all_agents_mu_flat_list, dim=-1)
            problem_ppo_data["all_sigmas_flat_for_inference"] = torch.cat(temp_all_agents_sigma_flat_list, dim=-1)
        else:
            total_expected_flat_latent_dim = 0
            for role_cfg_name_calc in ["planner", "coder", "debugger"]:
                role_config_calc = getattr(self.global_config, f"{role_cfg_name_calc}_role_config", None)
                total_agents_calc = getattr(self.global_config, f"total_{role_cfg_name_calc}_agents", 0)
                if role_config_calc and total_agents_calc > 0:
                    total_expected_flat_latent_dim += total_agents_calc * role_config_calc.N_ACTION_TEMPLATES * self.global_config.latent_dim
            problem_ppo_data["all_mus_flat_for_inference"] = torch.zeros(total_expected_flat_latent_dim, dtype=model_dtype, device='cpu')
            problem_ppo_data["all_sigmas_flat_for_inference"] = torch.zeros(total_expected_flat_latent_dim, dtype=model_dtype, device='cpu')

        if problem_ppo_data["all_z_for_diversity"]:
            problem_ppo_data["all_z_for_diversity"] = torch.stack(problem_ppo_data["all_z_for_diversity"], dim=0)
        else:
            problem_ppo_data["all_z_for_diversity"] = torch.empty(0, self.global_config.latent_dim, dtype=model_dtype, device='cpu')
        if not problem_ppo_data["all_mu_for_diversity"]:
             problem_ppo_data["all_mu_for_diversity"] = []
        if not problem_ppo_data["all_sigma_for_diversity"]:
             problem_ppo_data["all_sigma_for_diversity"] = []


        advantage_for_this_problem = problem_ppo_data["final_reward"] - problem_ppo_data["initial_value_prediction"].to(float32_dtype)
        for role_name_key_adv in problem_ppo_data["actor_inputs_by_role"]:
            num_turns_in_role = len(problem_ppo_data["actor_inputs_by_role"][role_name_key_adv]["obs_embs"])
            if num_turns_in_role > 0:
                problem_ppo_data["actor_inputs_by_role"][role_name_key_adv]["advantages"] = advantage_for_this_problem.repeat(num_turns_in_role).cpu()
            else:
                 problem_ppo_data["actor_inputs_by_role"][role_name_key_adv]["advantages"] = torch.empty(0, dtype=float32_dtype, device='cpu')

            for key in ["obs_embs", "h_prevs", "z_all_templates", "old_log_probs", "chosen_actions", "advantages"]:
                data_list_for_key = problem_ppo_data["actor_inputs_by_role"][role_name_key_adv][key]
                if isinstance(data_list_for_key, list):
                    if data_list_for_key:
                        problem_ppo_data["actor_inputs_by_role"][role_name_key_adv][key] = torch.stack(data_list_for_key, dim=0)
                    else:
                        role_cfg_empty = getattr(self.global_config, f"{role_name_key_adv}_role_config", None)
                        empty_dtype = model_dtype
                        if key in ["old_log_probs", "advantages"]: empty_dtype = float32_dtype
                        elif key == "chosen_actions": empty_dtype = torch.long

                        shape_empty = (0,)
                        if role_cfg_empty:
                            if key == "obs_embs": shape_empty = (0, role_cfg_empty.obs_embed_dim)
                            elif key == "z_all_templates": shape_empty = (0, role_cfg_empty.N_ACTION_TEMPLATES, self.global_config.latent_dim)
                        if key == "h_prevs": shape_empty = (0, self.global_config.actor_rnn_hidden_dim)
                        problem_ppo_data["actor_inputs_by_role"][role_name_key_adv][key] = torch.empty(shape_empty, dtype=empty_dtype, device='cpu')

        if problem_ppo_data["llm_generated_tokens_per_turn"]:
            tokens_to_pad = [t.squeeze(0) if t.ndim == 2 and t.size(0) == 1 else t for t in problem_ppo_data["llm_generated_tokens_per_turn"]]
            problem_ppo_data["llm_generated_tokens_per_turn"] = torch.nn.utils.rnn.pad_sequence(
                tokens_to_pad, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )
        else: problem_ppo_data["llm_generated_tokens_per_turn"] = torch.empty(0, 0, dtype=torch.long, device='cpu')

        if problem_ppo_data["llm_log_probs_per_turn"]:
            problem_ppo_data["llm_log_probs_per_turn"] = torch.stack(problem_ppo_data["llm_log_probs_per_turn"], dim=0)
        else: problem_ppo_data["llm_log_probs_per_turn"] = torch.empty(0, dtype=float32_dtype, device='cpu')

        logger.info(f"RUN_PIPELINE: [{problem_name}] Data collection and processing complete. Returning PPO data.")
        return problem_ppo_data

    def run(
        self, tokenizer: AutoTokenizer, llm_model: AutoModelForCausalLM,
        role_networks: Dict[str, Any], critic_net: torch.nn.Module,
        inference_net: torch.nn.Module, log: bool = True,
        num_problems_to_run: Optional[int] = None,
    ) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
        all_problem_eval_results: List[Dict[str, Any]] = []
        all_collected_ppo_data: List[Dict[str, Any]] = []
        problems_to_evaluate = self.tasks
        if num_problems_to_run is not None and num_problems_to_run > 0:
            problems_to_evaluate = self.tasks[:num_problems_to_run]
        logger.info(f"Starting pipeline evaluation for {len(problems_to_evaluate)} problems.")
        total_passed_tests, total_tests_run = 0, 0

        for problem_idx, task_item in enumerate(tqdm(problems_to_evaluate, desc="Running Pipeline for Problems")):
            h_crit_for_this_problem = torch.zeros(1, self.global_config.critic_rnn_hidden_dim, device=llm_model.device, dtype=self.model_dtype)
            collected_data = self.run_pipeline_for_problem(
                problem_idx, task_item, tokenizer, llm_model, role_networks,
                critic_net, inference_net, h_crit_for_this_problem
            )

            final_response_for_eval = ""
            if collected_data.get("llm_responses_per_turn"):
                final_response_for_eval = collected_data["llm_responses_per_turn"][-1]

            current_problem_eval_set = []
            if final_response_for_eval and not final_response_for_eval.startswith("ERROR_"):
                temp_filepath = self.sample_dir / f"final_eval_temp_{task_item['name'].replace('/', '_')}.py"
                clean_code = re.sub(r'```python', '', final_response_for_eval); clean_code = re.sub(r'```', '', clean_code).strip()
                with open(temp_filepath, 'w', encoding='utf-8') as f: f.write(clean_code)
                eval_data = (problem_idx, task_item, final_response_for_eval, str(temp_filepath))
                current_problem_eval_set = evaluate_single_problem(eval_data, self.code_pattern)
                if temp_filepath.exists(): os.remove(temp_filepath)

            all_problem_eval_results.extend(current_problem_eval_set)
            all_collected_ppo_data.append(collected_data)

            for res_item in current_problem_eval_set:
                if res_item['passed']: total_passed_tests += 1
            total_tests_run += len(current_problem_eval_set) if current_problem_eval_set else len(task_item.get("tests", []))

        if log and all_problem_eval_results:
            logger.info(f"Saving execution results to CSV file: {self.log_file}")
            with open(self.log_file, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["problem_name", "description", "test_case", "difficulty", "expected_output", "generated_output", "execution_error", "passed"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_problem_eval_results)

        avg_pass_rate = (total_passed_tests / total_tests_run * 100) if total_tests_run > 0 else 0.0
        logger.info(f"Average pass rate across all problems: {avg_pass_rate:.2f}% ({total_passed_tests}/{total_tests_run})")
        return avg_pass_rate, all_problem_eval_results, all_collected_ppo_data

if __name__ == "__main__":
    from marcog_config import GlobalSHPPOConfig
    from marcog import Encoder, LatentNet, ActorNet, CriticNet, InferenceNet

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("Running standalone marcog_test.py for basic CodeTester functionality check.")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = GlobalSHPPOConfig()
    cfg.device = device
    cfg.model_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32

    dataset = CodeContestDataset(split="test", max_problems=1)
    logger.info(f"Configured number of agents:")
    logger.info(f"  Planners: {cfg.total_planner_agents}")
    logger.info(f"  Coders: {cfg.total_coder_agents}")
    logger.info(f"  Debuggers: {cfg.total_debugger_agents}")
    tokenizer_main, llm_model_main = load_qwen_model(model_name=cfg.llm_model_name, model_dtype=cfg.model_dtype)
    cfg.update_llm_dims(llm_model_main.config.hidden_size, llm_model_main.dtype)


    role_networks_main = {}
    model_dtype_for_nets = cfg.model_dtype
    if cfg.total_planner_agents > 0 and cfg.planner_role_config:
        planner_encoder = Encoder(cfg.planner_role_config, cfg).to(device, dtype=model_dtype_for_nets)
        planner_latent_net = LatentNet(planner_encoder, cfg.planner_role_config, cfg).to(device, dtype=model_dtype_for_nets)
        planner_actor_net = ActorNet(cfg.planner_role_config, cfg).to(device, dtype=model_dtype_for_nets)
        role_networks_main["planner"] = {"actor_net": planner_actor_net, "latent_net": planner_latent_net}
    if cfg.total_coder_agents > 0 and cfg.coder_role_config:
        coder_encoder = Encoder(cfg.coder_role_config, cfg).to(device, dtype=model_dtype_for_nets)
        coder_latent_net = LatentNet(coder_encoder, cfg.coder_role_config, cfg).to(device, dtype=model_dtype_for_nets)
        coder_actor_net = ActorNet(cfg.coder_role_config, cfg).to(device, dtype=model_dtype_for_nets)
        role_networks_main["coder"] = {"actor_net": coder_actor_net, "latent_net": coder_latent_net}
    if cfg.total_debugger_agents > 0 and cfg.debugger_role_config:
        debugger_encoder = Encoder(cfg.debugger_role_config, cfg).to(device, dtype=model_dtype_for_nets)
        debugger_latent_net = LatentNet(debugger_encoder, cfg.debugger_role_config, cfg).to(device, dtype=model_dtype_for_nets)
        debugger_actor_net = ActorNet(cfg.debugger_role_config, cfg).to(device, dtype=model_dtype_for_nets)
        role_networks_main["debugger"] = {"actor_net": debugger_actor_net, "latent_net": debugger_latent_net}

    critic_net_main = CriticNet(cfg).to(device, dtype=model_dtype_for_nets)
    inference_net_main = InferenceNet(cfg).to(device, dtype=model_dtype_for_nets)

    tester = CodeTester(dataset=dataset, global_config=cfg, tokenizer=tokenizer_main, llm_model=llm_model_main, log_file="temp_standalone_test_log.csv")

    avg_rate, _, _ = tester.run(
        tokenizer=tokenizer_main, llm_model=llm_model_main, role_networks=role_networks_main,
        critic_net=critic_net_main, inference_net=inference_net_main, log=True, num_problems_to_run=1
    )
    print(f"Standalone test: Overall average pass rate: {avg_rate:.2f}%")