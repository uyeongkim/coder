"""
env.py - Environment and Dataset Management
No try-catch, throw errors directly
"""

import os
import re
import csv
import sys
import subprocess
import tempfile
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm, trange
from collections import Counter

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset

import logging
logger = logging.getLogger(__name__)


class CodeContestDataset:
    """CodeContest Dataset Class"""
    
    def __init__(self, split: str = "train", max_problems: int = -1, max_cases: int = -1):
        self.tasks = []
        self._load_dataset(split, max_problems, max_cases)
    
    def _load_dataset(self, split: str, max_problems: int, max_cases: int):
        """Load dataset"""
        logger.info(f"Loading {split} dataset...")
        
        if max_problems <= 0: 
            max_problems = float('inf')
        if max_cases <= 0: 
            max_cases = float('inf')
        
        # Load Hugging Face dataset
        ds = load_dataset("deepmind/code_contests", split=split, streaming=False)
        filtered_ds = ds.filter(self._is_valid_task)
        
        count = 0
        for row in tqdm(filtered_ds, desc=f"Loading {split} data", 
                      total=min(len(filtered_ds), max_problems)):
            if count >= max_problems:
                break
            
            task = self._process_task(row, max_cases)
            if task:
                self.tasks.append(task)
                count += 1
        
        logger.info(f"Loaded {len(self.tasks)} valid tasks")
    
    def _is_valid_task(self, row) -> bool:
        """Task validation"""
        # Check basic fields
        if not all(key in row for key in ['description', 'name', 'private_tests']):
            return False
        
        # Exclude file I/O tasks
        if row.get('input_file') or row.get('output_file'):
            return False
        
        # Check description length
        desc = row.get('description', '')
        if not isinstance(desc, str) or not (50 < len(desc) < 2000):
            return False
        
        # Check test cases
        private_tests = row.get('private_tests', {})
        if not private_tests.get('input') or not private_tests.get('output'):
            return False
        
        # Check test case count
        inputs = private_tests['input']
        outputs = private_tests['output']
        if len(inputs) != len(outputs) or len(inputs) == 0:
            return False
        
        return True
    
    def _process_task(self, row, max_cases: int) -> Optional[Dict[str, Any]]:
        """Process task data"""
        # Handle time limit
        time_limit = 2.0  # default
        if row.get('time_limit'):
            if isinstance(row['time_limit'], dict):
                if 'seconds' in row['time_limit']:
                    time_limit = float(row['time_limit']['seconds'])
                elif 'nanos' in row['time_limit']:
                    time_limit = float(row['time_limit']['nanos']) / 1e9
            elif isinstance(row['time_limit'], (int, float)):
                time_limit = float(row['time_limit'])
        
        # Limit time to reasonable range
        time_limit = max(1.0, min(time_limit, 10.0))
        
        # Process test cases
        inputs = row['private_tests']['input']
        outputs = row['private_tests']['output']
        
        test_count = min(len(inputs), max_cases)
        tests = []
        
        for i in range(test_count):
            inp = str(inputs[i]).strip()
            out = str(outputs[i]).strip()
            if inp or out:  # exclude empty test cases
                tests.append((inp, out))
        
        if not tests:
            return None
        
        # Handle difficulty
        difficulty = row.get('cf_rating', 1000)
        if not isinstance(difficulty, (int, float)):
            difficulty = 1000
        
        return {
            "name": str(row['name']).strip(),
            "prompt": str(row['description']).strip(),
            "difficulty": int(difficulty),
            "tests": tests,
            "time_limit": time_limit,
        }
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks


class SafeCodeExecutor:
    """Safe Code Execution Engine"""
    
    @staticmethod
    def extract_solve_function(response: str) -> Optional[str]:
        """Extract Python code from response"""
        # Python code block patterns
        patterns = [
            r'```python\s*(.*?)\s*```',
            r'```\s*(def solve.*?)\s*```',
            r'(def solve.*?)(?=\n\n|\nclass|\ndef(?!\s+solve)|\n#|\Z)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                code = matches[0].strip()
                if 'def solve' in code:
                    return code
        
        # If no code block, search in entire response
        if 'def solve' in response:
            return response.strip()
        
        return None
    
    @staticmethod
    def execute_code_safely(code: str, input_data: str, timeout: float) -> Tuple[bool, str, str]:
        """Safe code execution via multiprocessing"""
        def run_code(code_str, input_str, result_queue):
            """Execute code in separate process"""
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                full_code = f"""
import sys

# Original code
{code_str}

# Input data
input_data = '''{input_str}'''

# Check if solve function exists
if 'solve' not in globals():
    print("ERROR: solve function not found")
    sys.exit(1)

# Execute function
result = solve(input_data)

# Output result
if result is not None:
    print(str(result))
else:
    print("ERROR: solve function returned None")
    sys.exit(1)
"""
                f.write(full_code)
                temp_file = f.name
            
            # Execute via subprocess
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Delete temporary file
            os.unlink(temp_file)
            
            # Analyze result
            if process.returncode == 0:
                output = stdout.strip()
                if output.startswith("ERROR:"):
                    result_queue.put(('error', '', output))
                else:
                    result_queue.put(('success', output, ''))
            else:
                error_output = stdout.strip() if stdout.strip() else stderr.strip()
                result_queue.put(('error', '', error_output))
        
        # Multiprocessing execution
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_code, args=(code, input_data, result_queue))
        process.start()
        process.join(timeout + 2)
        
        # Force terminate if process still alive
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            return False, '', f'Process timeout after {timeout}s'
        
        # Get result
        if not result_queue.empty():
            status, output, error = result_queue.get_nowait()
            if status == 'success':
                return True, output, ''
            else:
                return False, '', error
        else:
            return False, '', 'No result from process'


class CodeTester:
    """Code Testing Environment"""
    
    def __init__(self, dataset, batch_size: int = 8, max_workers: int = 2, log_file: Optional[str] = None):
        self.tasks = dataset.get_all_tasks()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.log_file = log_file or "execution_log.csv"
        self.sample_dir = Path("generated_samples")
        self.sample_dir.mkdir(exist_ok=True)
    
    def generate_solutions(self, tokenizer, model, prompts: List[str], **kwargs) -> Union[List[str], Tuple]:
        """Generate solutions"""
        return_hidden = kwargs.get('return_hidden', False)
        return_logprobs = kwargs.get('return_logprobs', False)
        num_return_sequences = kwargs.get('num_return_seqs', 1)
        
        all_responses = []
        all_hidden_states = []
        all_tokens = []
        all_log_probs = []
        
        # Prompt template
        template = (
            "Write a Python function to solve the following programming problem.\n\n"
            "### Problem Description\n{}\n\n"
            "Function signature: def solve(input_str: str) -> str\n"
            "- input_str: input data (string)\n"
            "- return: answer (string)\n\n"
            "Only provide Python code wrapped in ```python``` markers."
        )
        
        # Batch processing
        for i in trange(0, len(prompts), self.batch_size, desc="Generating solutions"):
            batch_prompts = prompts[i:i + self.batch_size]
            
            # Apply chat template
            formatted_prompts = []
            for prompt in batch_prompts:
                messages = [
                    {"role": "system", "content": "You are an expert programmer."},
                    {"role": "user", "content": template.format(prompt)}
                ]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                formatted_prompts.append(formatted)
            
            # Tokenization
            inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            with torch.no_grad():
                # Extract hidden states
                if return_hidden:
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                    hidden_states = outputs.hidden_states[-1]
                    seq_lens = inputs.attention_mask.sum(dim=1) - 1
                    batch_hidden = hidden_states[torch.arange(hidden_states.size(0)), seq_lens, :]
                    all_hidden_states.append(batch_hidden.cpu())
                
                # Text generation
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    min_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=return_logprobs,
                    use_cache=True
                )
                
                # Extract generated tokens only
                prompt_len = inputs.input_ids.shape[1]
                generated_tokens = gen_outputs.sequences[:, prompt_len:]
                
                # Decode
                batch_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                all_responses.extend(batch_responses)
                
                # Calculate log probabilities
                if return_logprobs:
                    for seq in generated_tokens:
                        all_tokens.append(seq.cpu())
                    
                    if gen_outputs.scores:
                        # Simple log probability calculation
                        for i in range(len(generated_tokens)):
                            log_prob = torch.tensor(-1.0)  # default value
                            if gen_outputs.scores and len(gen_outputs.scores) > 0:
                                first_score = gen_outputs.scores[0][i]
                                probs = torch.softmax(first_score, dim=-1)
                                if len(generated_tokens[i]) > 0:
                                    token_id = generated_tokens[i][0]
                                    log_prob = torch.log(probs[token_id] + 1e-8)
                            all_log_probs.append(log_prob)
                    else:
                        all_log_probs.extend([torch.tensor(-1.0) for _ in range(len(generated_tokens))])
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Construct return value
        result = [all_responses]
        
        if return_hidden and all_hidden_states:
            result.append(torch.cat(all_hidden_states, dim=0))
        elif return_hidden:
            result.append(torch.empty(0))
        
        if return_logprobs:
            if all_tokens:
                # Token padding
                max_len = max(t.size(0) for t in all_tokens) if all_tokens else 1
                padded_tokens = []
                for token_seq in all_tokens:
                    if token_seq.size(0) < max_len:
                        padding = torch.full((max_len - token_seq.size(0),), tokenizer.pad_token_id)
                        padded_seq = torch.cat([token_seq, padding])
                    else:
                        padded_seq = token_seq[:max_len]
                    padded_tokens.append(padded_seq)
                result.append(torch.stack(padded_tokens))
            else:
                result.append(torch.empty(0))
            
            if all_log_probs:
                result.append(torch.stack(all_log_probs))
            else:
                result.append(torch.empty(0))
        
        return tuple(result) if len(result) > 1 else result[0]
    
    def evaluate_solution(self, task_and_solution: Tuple[Dict[str, Any], str]) -> List[Dict[str, Any]]:
        """Evaluate single solution with detailed error classification"""
        task, solution = task_and_solution
        problem_name = task["name"]
        tests = task["tests"]
        timeout = task["time_limit"]
        difficulty = task.get("difficulty", 1000)  # Get difficulty for logging
        
        # Extract code
        code = SafeCodeExecutor.extract_solve_function(solution)
        if not code:
            return [{
                "problem_name": problem_name,
                "test_case": inp,
                "expected_output": expected,
                "generated_output": None,
                "execution_error": "solve function not found",
                "error_type": "CODE_EXTRACTION_ERROR",
                "difficulty": difficulty,
                "passed": False,
            } for inp, expected in tests]
        
        results = []
        for test_idx, (inp, expected) in enumerate(tests):
            # Execute code
            success, output, error = SafeCodeExecutor.execute_code_safely(code, inp, timeout)
            
            # Initialize result variables
            passed = False
            generated_output = None
            execution_error = ""
            error_type = "NONE"
            
            if success:
                # Code executed successfully
                if output is not None and output.strip():
                    generated_output = str(output).strip()
                    expected_output = str(expected).strip()
                    passed = generated_output == expected_output
                    
                    if not passed:
                        error_type = "WRONG_ANSWER"
                        execution_error = f"WRONG_ANSWER: Expected '{expected_output}', got '{generated_output}'"
                else:
                    # Success but no output
                    generated_output = ""
                    error_type = "NO_OUTPUT"
                    execution_error = "NO_OUTPUT: Function executed but produced no output"
            else:
                # Code execution failed
                generated_output = None
                
                # Classify error types
                error_lower = error.lower()
                
                if "timeout" in error_lower or "time" in error_lower:
                    error_type = "TIMEOUT"
                    execution_error = f"TIMEOUT: {error}"
                elif "no output produced" in error_lower:
                    error_type = "NO_OUTPUT"
                    execution_error = f"NO_OUTPUT: {error}"
                elif "solve function not found" in error_lower:
                    error_type = "FUNCTION_NOT_FOUND"
                    execution_error = f"FUNCTION_NOT_FOUND: {error}"
                elif "syntaxerror" in error_lower or "syntax" in error_lower:
                    error_type = "SYNTAX_ERROR"
                    execution_error = f"SYNTAX_ERROR: {error}"
                elif "nameerror" in error_lower:
                    error_type = "NAME_ERROR"
                    execution_error = f"NAME_ERROR: {error}"
                elif "typeerror" in error_lower:
                    error_type = "TYPE_ERROR"
                    execution_error = f"TYPE_ERROR: {error}"
                elif "valueerror" in error_lower:
                    error_type = "VALUE_ERROR"
                    execution_error = f"VALUE_ERROR: {error}"
                elif "indexerror" in error_lower:
                    error_type = "INDEX_ERROR"
                    execution_error = f"INDEX_ERROR: {error}"
                elif "keyerror" in error_lower:
                    error_type = "KEY_ERROR"
                    execution_error = f"KEY_ERROR: {error}"
                elif "attributeerror" in error_lower:
                    error_type = "ATTRIBUTE_ERROR"
                    execution_error = f"ATTRIBUTE_ERROR: {error}"
                elif "zerodivisionerror" in error_lower:
                    error_type = "ZERO_DIVISION_ERROR"
                    execution_error = f"ZERO_DIVISION_ERROR: {error}"
                elif "recursionerror" in error_lower or "maximum recursion" in error_lower:
                    error_type = "RECURSION_ERROR"
                    execution_error = f"RECURSION_ERROR: {error}"
                elif "memoryerror" in error_lower:
                    error_type = "MEMORY_ERROR"
                    execution_error = f"MEMORY_ERROR: {error}"
                elif "importerror" in error_lower or "modulenotfounderror" in error_lower:
                    error_type = "IMPORT_ERROR"
                    execution_error = f"IMPORT_ERROR: {error}"
                elif "indentationerror" in error_lower:
                    error_type = "INDENTATION_ERROR"
                    execution_error = f"INDENTATION_ERROR: {error}"
                elif "runtimeerror" in error_lower:
                    error_type = "RUNTIME_ERROR"
                    execution_error = f"RUNTIME_ERROR: {error}"
                elif "process timeout" in error_lower:
                    error_type = "PROCESS_TIMEOUT"
                    execution_error = f"PROCESS_TIMEOUT: {error}"
                elif "no result" in error_lower:
                    error_type = "NO_RESULT"
                    execution_error = f"NO_RESULT: {error}"
                else:
                    error_type = "UNKNOWN_ERROR"
                    execution_error = f"UNKNOWN_ERROR: {error}"
            
            results.append({
                "problem_name": problem_name,
                "test_case": inp,
                "expected_output": expected,
                "generated_output": generated_output,
                "execution_error": execution_error,
                "error_type": error_type,
                "difficulty": difficulty,
                "passed": passed,
            })
        
        return results
    
    def run(self, tokenizer, model, return_hidden: bool = False, 
            return_logprobs: bool = False, log: bool = True, 
            num_return_seqs: int = 1) -> Union[float, Tuple]:
        """Main execution function"""
        
        print(f"CodeTester running: {len(self.tasks)} problems")
        
        # Extract prompts
        prompts = [task["prompt"] for task in self.tasks]
        
        # Generate solutions
        print("Generating AI solutions...")
        generation_result = self.generate_solutions(
            tokenizer, model, prompts,
            return_hidden=return_hidden,
            return_logprobs=return_logprobs,
            num_return_seqs=num_return_seqs
        )
        
        # Unpack results
        if return_hidden and return_logprobs:
            solutions, hidden_states, tokens, log_probs = generation_result
        elif return_hidden:
            solutions, hidden_states = generation_result
            tokens, log_probs = None, None
        elif return_logprobs:
            solutions, tokens, log_probs = generation_result
            hidden_states = None
        else:
            solutions = generation_result
            hidden_states, tokens, log_probs = None, None, None
        
        print(f"Generated {len(solutions)} solutions")
        
        # Save solutions
        for i, (task, solution) in enumerate(zip(self.tasks, solutions)):
            self._save_solution(task["name"], solution, i)
        
        # Execute and evaluate code
        print("Executing and evaluating solutions...")
        all_results = []
        
        # Create task-solution pairs
        task_solution_pairs = list(zip(self.tasks, solutions))
        
        # Parallel execution
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.evaluate_solution, pair)
                for pair in task_solution_pairs
            ]
            
            for future in tqdm(as_completed(futures, timeout=300), 
                             total=len(futures), desc="Code execution and evaluation"):
                results = future.result(timeout=60)  # max 1 minute per problem
                all_results.extend(results)
        
        # Save results
        if log and all_results:
            self._save_results(all_results)
        
        # Calculate statistics
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r["passed"])
        error_tests = sum(1 for r in all_results if r["execution_error"])
        
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
        error_rate = (error_tests / total_tests * 100) if total_tests > 0 else 0.0
        
        print(f"Evaluation complete:")
        print(f"  - Total tests: {total_tests}")
        print(f"  - Passed: {passed_tests} ({pass_rate:.1f}%)")
        print(f"  - Execution errors: {error_tests} ({error_rate:.1f}%)")
        
        # Return values
        if return_hidden and return_logprobs:
            return pass_rate, solutions, hidden_states, tokens, log_probs
        elif return_hidden:
            return pass_rate, solutions, hidden_states
        elif return_logprobs:
            return pass_rate, solutions, tokens, log_probs
        else:
            return pass_rate
    
    def _save_solution(self, task_name: str, solution: str, idx: int):
        """Save solution to file"""
        # Generate safe filename
        safe_name = re.sub(r'[^\w\-_\.]', '_', task_name.strip())
        if not safe_name:
            safe_name = f"task_{idx}"
        
        filepath = self.sample_dir / f"{safe_name}.py"
        counter = 1
        while filepath.exists():
            filepath = self.sample_dir / f"{safe_name}_{counter}.py"
            counter += 1
        
        # Clean code
        clean_solution = solution
        clean_solution = re.sub(r'```python\s*', '', clean_solution)
        clean_solution = re.sub(r'```\s*', '', clean_solution)
        clean_solution = clean_solution.strip()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Task: {task_name}\n")
            f.write(f"# Generated solution\n\n")
            f.write(clean_solution)
            f.write("\n")
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save results to CSV"""
        fieldnames = [
            "problem_name", "test_case", "expected_output",
            "generated_output", "execution_error", "error_type", "difficulty", "passed"
        ]
        
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Select necessary fields and handle None values
                row = {}
                for field in fieldnames:
                    value = result.get(field, "")
                    if value is None:
                        value = ""
                    # Truncate large text
                    if isinstance(value, str) and len(value) > 1000:
                        value = value[:1000] + "..."
                    row[field] = value
                writer.writerow(row)
        
        # Print error statistics
        self._print_error_statistics(results)
        print(f"Results saved to {self.log_file}")
    
    def _print_error_statistics(self, results: List[Dict[str, Any]]):
        """Print error statistics"""
        # Error type statistics
        error_types = [r.get('error_type', 'NONE') for r in results]
        error_counts = Counter(error_types)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get('passed', False))
        
        print(f"\n=== Execution Result Statistics ===")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Failed: {total_tests - passed_tests} ({(total_tests-passed_tests)/total_tests*100:.1f}%)")
        
        print(f"\n=== Error Type Statistics ===")
        for error_type, count in error_counts.most_common():
            percentage = count / total_tests * 100
            print(f"{error_type}: {count} ({percentage:.1f}%)")
        
        # Descriptions for common errors
        common_errors = {
            'WRONG_ANSWER': 'Wrong answer (logic error)',
            'NO_OUTPUT': 'No output produced',
            'TIMEOUT': 'Execution timeout',
            'SYNTAX_ERROR': 'Syntax error',
            'NAME_ERROR': 'Undefined variable/function',
            'TYPE_ERROR': 'Type-related error',
            'VALUE_ERROR': 'Value-related error',
            'INDEX_ERROR': 'Index out of range',
            'RECURSION_ERROR': 'Recursion limit exceeded',
            'RUNTIME_ERROR': 'Runtime error'
        }
        
        print(f"\n=== Main Error Descriptions ===")
        for error_type, count in error_counts.most_common(5):
            if error_type in common_errors and count > 0:
                print(f"{error_type}: {common_errors[error_type]}")


def test_code_execution():
    """Test code execution with various scenarios"""
    print("Testing code execution with multiple scenarios...")
    
    test_cases = [
        {
            "name": "Simple Addition",
            "code": """
def solve(input_str):
    lines = input_str.strip().split('\\n')
    a, b = map(int, lines[0].split())
    return str(a + b)
""",
            "input": "3 5",
            "expected": "8"
        },
        {
            "name": "No Output (None return)",
            "code": """
def solve(input_str):
    a, b = map(int, input_str.strip().split())
    result = a + b
    return None  # This should fail
""",
            "input": "3 5",
            "expected": "8"
        },
        {
            "name": "Empty String Output",
            "code": """
def solve(input_str):
    return ""  # This should fail
""",
            "input": "3 5",
            "expected": "8"
        },
        {
            "name": "Syntax Error",
            "code": """
def solve(input_str):
    a, b = map(int, input_str.strip().split())
    return str(a + b  # Missing closing parenthesis
""",
            "input": "3 5",
            "expected": "8"
        },
        {
            "name": "Runtime Error",
            "code": """
def solve(input_str):
    a, b = map(int, input_str.strip().split())
    return str(a / 0)  # Division by zero
""",
            "input": "3 5",
            "expected": "8"
        }
    ]
    
    print(f"Running {len(test_cases)} test cases...")
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        
        success, output, error = SafeCodeExecutor.execute_code_safely(
            test_case['code'], 
            test_case['input'], 
            5.0
        )
        
        expected_success = test_case['name'] == "Simple Addition"
        
        print(f"  Success: {success}")
        print(f"  Output: '{output}'")
        print(f"  Error: '{error}'")
        print(f"  Expected to succeed: {expected_success}")
        
        if expected_success:
            test_passed = success and output.strip() == test_case['expected']
        else:
            test_passed = not success and error  # Should fail with an error message
        
        print(f"  Test result: {'PASS' if test_passed else 'FAIL'}")
        results.append(test_passed)
    
    all_passed = all(results)
    print(f"\nOverall test result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    return all_passed


if __name__ == "__main__":
    # Test code execution
    if test_code_execution():
        print("Code execution test passed!")
    else:
        print("Code execution test failed!")