# shppo_env.py
import os
import sys
from pathlib import Path
import re
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional, Callable
import torch
from datasets import load_dataset
import logging
import concurrent.futures
from functools import partial
import csv
import numpy as np
import random

from shppo_config import SHPPOConfig

logger = logging.getLogger(__name__)

class CodeContestDataset:
    """Loads and processes tasks from the deepmind/code_contests dataset."""
    def __init__(self, split: str = "train", max_problems: int = -1, max_cases: int = -1, cache_dir: Optional[str] = None):
        def _is_valid(row: Optional[Dict]) -> bool:
            if row is None: return False
            if row.get('input_file') or row.get('output_file'): return False
            if row.get('description') is None or not (10 < len(row['description']) < 10000): return False
            if row.get('name') is None or row.get('public_tests') is None or not row['public_tests'].get('input'): return False
            if row.get('solutions') is None: return False
            if row.get('time_limit') is None or not (row['time_limit'].get('seconds') or row['time_limit'].get('nanos')): return False
            return True
        if max_problems <= 0: max_problems = int(1e9)
        if max_cases <= 0: max_cases = int(1e9)
        logger.info(f"Loading CodeContestDataset {split} (max_problems={max_problems}, max_cases={max_cases})")
        self.tasks: List[Dict[str, Any]] = []
        try: ds = load_dataset("deepmind/code_contests", split=split, trust_remote_code=False, cache_dir=cache_dir)
        except Exception as e: logger.error(f"Failed to load dataset 'deepmind/code_contests': {e}"); raise
        ds_filtered = ds.filter(_is_valid); count = 0
        for row in tqdm(ds_filtered, desc=f"Processing {split} tasks for CodeContestDataset"):
            if count >= max_problems: break
            name, desc = row['name'], row['description']
            public_tests_data = row['public_tests']; ins, outs = public_tests_data.get('input', []), public_tests_data.get('output', [])
            time_limit_data = row['time_limit']; time_limit_seconds: float = 0.0
            if 'seconds' in time_limit_data and time_limit_data['seconds'] is not None: time_limit_seconds = float(time_limit_data['seconds'])
            elif 'nanos' in time_limit_data and time_limit_data['nanos'] is not None: time_limit_seconds = float(time_limit_data['nanos']) / 1e9
            if not ins: continue
            self.tasks.append({"name": name, "prompt": desc, "tests_public": list(zip(ins, outs))[:max_cases], "time_limit": time_limit_seconds if time_limit_seconds > 0 else 10.0})
            count += 1
        logger.info(f"Loaded {len(self.tasks)} valid tasks from {split} for CodeContestDataset.")
        if not self.tasks: logger.warning(f"No tasks were loaded for CodeContestDataset (split {split}).")
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Returns all loaded and processed tasks."""
        return self.tasks

def extract_code_from_response(response: str, code_pattern: Optional[re.Pattern] = None) -> str:
    if code_pattern is None: code_pattern = re.compile(r'```python\s*([\s\S]*?)\s*```', re.IGNORECASE)
    match = code_pattern.search(response)
    if match: return match.group(1).strip()
    if 'def solve' in response:
        lines = response.split('\n'); code_lines: List[str] = []
        in_solve_block: bool = False; indent_level: int = -1
        for ln, line_content in enumerate(lines):
            sl = line_content.lstrip(); ci = len(line_content) - len(sl)
            if 'def solve' in sl and not in_solve_block: in_solve_block, indent_level = True, ci
            if in_solve_block:
                if ci < indent_level and sl and ln > 0:
                    if ln + 1 < len(lines):
                        if not lines[ln+1].lstrip() or (len(lines[ln+1]) - len(lines[ln+1].lstrip()) < indent_level): break
                    else: break
                code_lines.append(line_content)
        if code_lines: return "\n".join(code_lines).strip()
    return response

def extract_executable_solve_function(code_str: str) -> Optional[Callable]:
    if not code_str or 'def solve' not in code_str: return None
    namespace: Dict[str, Any] = {}
    try: exec(code_str, namespace); return namespace.get("solve")
    except Exception as e: logger.debug(f"Code exec error for solve fn: {e}\nCode: {code_str[:200]}"); return None

def _execute_solve_fn_with_timeout(solve_fn: Callable, inp: Any, timeout_seconds: float) -> Any:
    if timeout_seconds <= 0: return solve_fn(inp)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(solve_fn, inp)
        try: return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError: raise TimeoutError(f"Execution timed out after {timeout_seconds}s.")

def evaluate_code_for_csv(
    problem_name: str, task_description: str, generated_code_str: str,
    tests_public: List[Tuple[Any, Any]], time_limit: float,
    code_extraction_fn: Callable[[str], str],
    solve_fn_extraction_fn: Callable[[str], Optional[Callable]]
) -> List[Dict[str, Any]]:
    results_for_csv: List[Dict[str, Any]] = []
    extracted_code = code_extraction_fn(generated_code_str)
    if not extracted_code or "def solve" not in extracted_code:
        error_msg = "Failed to extract valid code with 'def solve'"
        for inp, exp_out in tests_public: results_for_csv.append({"problem_name": problem_name, "description": task_description, "test_case_input": inp, "expected_output": exp_out, "generated_output": None, "execution_error": error_msg, "passed": False})
        return results_for_csv
    solve_fn = solve_fn_extraction_fn(extracted_code)
    if solve_fn is None:
        error_msg = "Failed to compile solve function"
        for inp, exp_out in tests_public: results_for_csv.append({"problem_name": problem_name, "description": task_description, "test_case_input": inp, "expected_output": exp_out, "generated_output": None, "execution_error": error_msg, "passed": False})
        return results_for_csv
    for inp, exp_out in tests_public:
        actual_out_str, error_str, is_passed = None, "", False
        try:
            actual_out = _execute_solve_fn_with_timeout(solve_fn, inp, time_limit)
            actual_out_str = str(actual_out).strip()
            if actual_out_str == str(exp_out).strip(): is_passed = True
        except TimeoutError as e_timeout: error_str = f"TimeoutError:{str(e_timeout)[:100]}"
        except Exception as e_runtime: error_str = f"RuntimeError:{str(e_runtime)[:100]}"
        results_for_csv.append({"problem_name": problem_name, "description": task_description, "test_case_input": inp, "expected_output": exp_out, "generated_output": actual_out_str, "execution_error": error_str, "passed": is_passed})
    return results_for_csv


class SHPPOCodeEnv:
    """
    Handles sequential Multi-Agent Reinforcement Learning (MARL) for code generation tasks
    within multiple parallel environments.
    """
    def __init__(self, config: SHPPOConfig, all_problem_tasks: List[Dict]):
        self.cfg = config
        self.all_problem_tasks = all_problem_tasks
        if not self.all_problem_tasks: raise ValueError("No problem tasks provided.")
        self.num_envs = config.num_envs
        self.num_marl_agents = config.num_marl_agents
        self.env_states: List[Dict[str, Any]] = [self._create_empty_env_state() for _ in range(self.num_envs)]
        self.code_pattern_for_extraction = re.compile(r'```python\s*([\s\S]*?)\s*```', re.IGNORECASE)
        logger.info(f"SHPPOCodeEnv: {self.num_envs} parallel envs, {self.num_marl_agents} MARL agents/env (acting sequentially).")

    def _create_empty_env_state(self) -> Dict[str, Any]:
        """Helper to create a default initial state for one parallel environment."""
        return {
            "task_data": None, "current_prompt": "", "team_current_pass_fraction": 0.0,
            "team_overall_code": "", "team_plan": "", "team_errors_summary": "",
            "episode_steps": 0, "episode_done": False, "current_marl_agent_turn": 0,
            "marl_agents_last_actions_str": [self.cfg.ACTION_TEMPLATES[-1]] * self.num_marl_agents,
            "marl_agents_llm_responses_this_turn": ["" for _ in range(self.num_marl_agents)]
        }

    def _get_current_marl_agent_local_obs_components(self, env_idx: int) -> Dict[str, Any]:
        """
        Gets local observation components for the MARL agent whose turn it is in the specified environment.
        The observation depends on what previous agents in the sequence have done in THIS team step
        and the overall team state from PREVIOUS team steps.
        """
        env_s = self.env_states[env_idx]
        marl_agent_idx = env_s["current_marl_agent_turn"]
        return {
            "prompt": env_s.get("current_prompt", ""),
            "team_overall_code": env_s.get("team_overall_code", ""),
            "team_plan": env_s.get("team_plan", ""),
            "team_errors_summary": env_s.get("team_errors_summary", ""),
            "team_pass_fraction": env_s.get("team_current_pass_fraction", 0.0),
            "my_id": marl_agent_idx, # Agent's ID within the team (0 to num_marl_agents-1)
            "task_data": env_s.get("task_data", {}), # Contains problem name, original prompt, tests
            "episode_step": env_s.get("episode_steps", 0), # Current team step in the episode
            "my_turn_in_team_step": marl_agent_idx, # Explicitly stating agent's turn order
            "my_last_action_str": env_s["marl_agents_last_actions_str"][marl_agent_idx]
        }
    
    def _get_team_global_state_components(self, env_idx: int) -> Dict[str, Any]:
        """Gets global state components for the team in the specified environment."""
        env_s = self.env_states[env_idx]
        return {
            "team_pass_fraction": env_s.get("team_current_pass_fraction", 0.0),
            "team_errors_summary": env_s.get("team_errors_summary", ""),
            "episode_steps": env_s.get("episode_steps", 0),
            "num_active_marl_agents": self.num_marl_agents,
            "current_marl_agent_turn_global": env_s.get("current_marl_agent_turn", 0) # Could be used by critic/inference
        }

    def reset_all_envs(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """ 
        Resets all parallel environments.
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            - A list of initial local observation components, one for the *first* MARL agent (turn 0) in each environment.
            - A list of initial global state components, one for each environment.
        """
        all_envs_initial_first_agent_loc_obs_comps: List[Dict[str,Any]] = []
        all_envs_initial_global_state_comps: List[Dict[str,Any]] = []
        num_available_tasks = len(self.all_problem_tasks)
        if num_available_tasks == 0: raise ValueError("No tasks available for reset.")
        
        sampled_indices = []
        if num_available_tasks < self.num_envs:
            sampled_indices = np.random.choice(num_available_tasks, self.num_envs, replace=True).tolist()
        else:
            sampled_indices = random.sample(range(num_available_tasks), self.num_envs)

        for i in range(self.num_envs):
            problem_task = self.all_problem_tasks[sampled_indices[i]]
            first_agent_loc_obs, global_state = self._reset_one_env(i, problem_task)
            all_envs_initial_first_agent_loc_obs_comps.append(first_agent_loc_obs)
            all_envs_initial_global_state_comps.append(global_state)
        return all_envs_initial_first_agent_loc_obs_comps, all_envs_initial_global_state_comps

    def _reset_one_env(self, env_idx: int, problem_task: Dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets a single environment and returns observations for the first agent."""
        self.env_states[env_idx] = self._create_empty_env_state()
        self.env_states[env_idx].update({
            "task_data": problem_task, 
            "current_prompt": problem_task['prompt'],
            "current_marl_agent_turn": 0 # Ensure turn starts at 0
        })
        first_agent_loc_obs_comps = self._get_current_marl_agent_local_obs_components(env_idx)
        global_state_comps = self._get_team_global_state_components(env_idx)
        return first_agent_loc_obs_comps, global_state_comps

    def evaluate_one_marl_agent_code(self,
        task_data: Dict[str, Any],
        generated_code_str: str,
        code_extraction_fn: Callable[[str], str] = extract_code_from_response, 
        solve_fn_extraction_fn: Callable[[str], Optional[Callable]] = extract_executable_solve_function
    ) -> Tuple[float, str, bool]:
        """Evaluates a given code string, returning pass fraction, error message, and solved status."""
        if not task_data or 'tests_public' not in task_data:
            logger.warning("Task data or public tests missing for evaluation in evaluate_one_marl_agent_code.")
            return 0.0, "Task data or public tests missing", False

        tests_public = task_data['tests_public']
        time_limit = task_data.get('time_limit', 10.0)

        if not tests_public:
            logger.info("No public tests available for this task in evaluate_one_marl_agent_code.")
            return 0.0, "No public tests available", False 

        extracted_code = code_extraction_fn(generated_code_str)
        if not extracted_code or "def solve" not in extracted_code :
            return 0.0, "Failed to extract valid code with 'def solve'", False

        solve_fn = solve_fn_extraction_fn(extracted_code)
        if solve_fn is None:
            return 0.0, "Failed to compile solve function from extracted code", False

        passed_count = 0
        error_messages: List[str] = []

        for inp, expected_out in tests_public:
            try:
                actual_out = _execute_solve_fn_with_timeout(solve_fn, inp, time_limit)
                if str(actual_out).strip() == str(expected_out).strip():
                    passed_count += 1
                else:
                    error_messages.append(f"WA - Input: {str(inp)[:30]}..., Expected: {str(expected_out)[:30]}..., Got: {str(actual_out)[:30]}...")
            except TimeoutError:
                error_messages.append(f"Timeout - Input: {str(inp)[:30]}...")
            except Exception as e:
                error_messages.append(f"RuntimeError: {str(e)[:50]} - Input: {str(inp)[:30]}...")

        pass_fraction = (passed_count / len(tests_public)) if tests_public else 0.0
        solved = (abs(pass_fraction - 1.0) < 1e-9) 

        error_summary = ""
        if not solved:
            if error_messages:
                error_summary = "; ".join(error_messages[:2]) 
            elif pass_fraction < 1.0 and not error_messages and len(tests_public) > 0 :
                error_summary = f"{passed_count}/{len(tests_public)} tests passed but some failed without explicit error messages."
            elif len(tests_public) == 0:
                error_summary = "No public tests to evaluate."
        else:
            error_summary = "AllTestsPassed"
        return pass_fraction, error_summary, solved
    

    def step_agent_turn(self, env_idx: int, marl_agent_action_str: str, marl_agent_llm_response: str
                 ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], float, bool, bool, Dict]:
        """
        Processes an action for the MARL agent whose turn it currently is in the specified environment.
        Updates the environment state based on this single agent's action and advances the turn.
        If the agent is the last in the sequence for the current team step, the team's overall code is evaluated,
        and rewards/done status for the team step are determined.

        Args:
            env_idx: Index of the parallel environment.
            marl_agent_action_str: The action template string chosen by the current MARL agent.
            marl_agent_llm_response: The LLM response generated based on the chosen action.

        Returns:
            A tuple containing:
            - next_agent_loc_obs_comps_or_None (Optional[Dict[str, Any]]): 
                Local observation components for the *next* agent to act. 
                If the team step just ended, this is for the *first* agent of the *next* team step.
                None if the episode is done.
            - current_team_global_state_comps_or_None (Optional[Dict[str, Any]]): 
                Global state components for the team. This is typically relevant after a full team step,
                or can represent the evolving global state. None if episode is done.
            - team_reward (float): 
                The reward for the team. Non-zero only if a full team step has just concluded.
            - team_done (bool): 
                True if the episode has ended for the team (either solved or max steps reached).
                Determined only after a full team step.
            - is_turn_for_next_marl_agent_in_team (bool): 
                True if there is another MARL agent to act in the current team step.
                False if the team step has just concluded (all agents have acted).
            - info (Dict): 
                Additional information, e.g., pass fraction, error summary after a team step.
        """
        env_s = self.env_states[env_idx]
        if env_s.get("episode_done", False):
            return None, None, 0.0, True, False, {"info": "Episode was already done."}

        current_turn_agent_idx = env_s["current_marl_agent_turn"]
        env_s["marl_agents_last_actions_str"][current_turn_agent_idx] = marl_agent_action_str
        env_s["marl_agents_llm_responses_this_turn"][current_turn_agent_idx] = marl_agent_llm_response
        
        if marl_agent_action_str == "plan-subgoal": env_s["team_plan"] = marl_agent_llm_response
        elif marl_agent_action_str in ["generate-code", "patch-bug", "unit-fix", "optimize-code"]:
            extracted_code = extract_code_from_response(marl_agent_llm_response, self.code_pattern_for_extraction)
            if "def solve" in extracted_code: env_s["team_overall_code"] = extracted_code
        
        env_s["current_marl_agent_turn"] += 1
        is_team_step_over = env_s["current_marl_agent_turn"] >= self.num_marl_agents

        team_reward_for_this_turn = 0.0
        team_done_for_this_turn = False
        info = {}
        next_agent_loc_obs = None
        current_team_glob_state = self._get_team_global_state_components(env_idx) # Global state can be observed at any point

        if is_team_step_over:
            env_s["episode_steps"] += 1
            previous_team_pf = env_s["team_current_pass_fraction"]
            
            final_pf, final_err, solved = self.evaluate_one_marl_agent_code(
                env_s["task_data"], 
                env_s["team_overall_code"],
                code_extraction_fn=lambda r: extract_code_from_response(r, self.code_pattern_for_extraction),
                solve_fn_extraction_fn=extract_executable_solve_function
            )

            env_s["team_current_pass_fraction"] = final_pf
            # Update error summary only if there's a new meaningful error, not "AllTestsPassed"
            if final_err and not final_err.startswith("AllTestsPassed"):
                env_s["team_errors_summary"] = final_err
            elif solved: # If solved, clear previous errors.
                 env_s["team_errors_summary"] = "AllTestsPassed"


            team_reward_for_this_turn = (final_pf - previous_team_pf) * 1.0 # Reward for improvement
            if final_pf == 1.0 and previous_team_pf < 1.0: team_reward_for_this_turn += 1.0 # Bonus for solving
            elif final_pf < previous_team_pf: team_reward_for_this_turn -= 0.1 # Penalty for regression

            team_done_for_this_turn = solved or (env_s["episode_steps"] >= self.cfg.max_team_episode_steps)
            env_s["episode_done"] = team_done_for_this_turn
            
            env_s["current_marl_agent_turn"] = 0 
            env_s["marl_agents_llm_responses_this_turn"] = ["" for _ in range(self.num_marl_agents)] 
            
            if not team_done_for_this_turn:
                next_agent_loc_obs = self._get_current_marl_agent_local_obs_components(env_idx) # Obs for agent 0 of next team step
            # Global state is updated based on the completed team step
            current_team_glob_state = self._get_team_global_state_components(env_idx) 

            info = {"team_pass_fraction": final_pf, "team_error": env_s["team_errors_summary"], "solved": solved}
            return next_agent_loc_obs, current_team_glob_state, team_reward_for_this_turn, team_done_for_this_turn, False, info
        else:
            # Team step not over, get obs for the *next* agent in sequence
            next_agent_loc_obs = self._get_current_marl_agent_local_obs_components(env_idx)
            # No team reward/done until team step is over
            return next_agent_loc_obs, current_team_glob_state, 0.0, False, True, {}

    def run_evaluation_and_save_csv(self, problem_task_pairs_with_final_team_code: List[Tuple[Dict, str]], csv_filepath: str):
        """Runs evaluation on provided codes and saves results to a CSV file."""
        logger.info(f"Evaluation: Saving {len(problem_task_pairs_with_final_team_code)} team codes to {csv_filepath}")
        all_results:List[Dict[str,Any]] = []
        for task,code_str in tqdm(problem_task_pairs_with_final_team_code,desc="Evaluating Final Team Codes"):
            if not task: # Skip if task data is missing
                logger.warning("Skipping evaluation for an entry due to missing task data.")
                continue
            res = evaluate_code_for_csv(task['name'],task['prompt'],code_str,task['tests_public'],task['time_limit'],
                                        lambda r: extract_code_from_response(r,self.code_pattern_for_extraction),
                                        extract_executable_solve_function)
            all_results.extend(res)
        if not all_results: logger.warning("No results to save to CSV."); return 0.0
        try:
            with open(csv_filepath,"w",newline="",encoding="utf-8") as f:
                fn = ["problem_name","description","test_case_input","expected_output","generated_output","execution_error","passed"]
                w=csv.DictWriter(f,fieldnames=fn);w.writeheader();w.writerows(all_results)
            logger.info(f"Execution results saved to {csv_filepath}")
        except IOError as e: logger.error(f"Failed to write CSV {csv_filepath}:{e}"); return 0.0
        passed=sum(1 for r in all_results if r["passed"]);total=len(all_results)
        avg_pass=(passed/total*100) if total>0 else 0.0
        logger.info(f"Batch Eval: Avg Pass Rate={avg_pass:.2f}% ({passed}/{total})")
        return avg_pass