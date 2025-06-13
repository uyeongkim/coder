"""
yy_env.py - Environment & Reward System
성능 최적화: 병렬 처리, 캐싱, 메모리 관리
HumanEval + CodeContest Mixed Curriculum Learning
"""

from __future__ import annotations
import os
import re
import sys
import tempfile
import subprocess
import multiprocessing
import random
import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from datasets import load_dataset
import functools
import hashlib
import signal
import logging
from rich.logging import RichHandler


# ═══════════════════════════════════════════════════════════════════════════
# Logger Setup
# ═══════════════════════════════════════════════════════════════════════════
RICH_FORMAT = "%(message)s"

logging.basicConfig(
    level="INFO",
    format=RICH_FORMAT,
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.exit(0)
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
sys.excepthook = handle_exception

# ═══════════════════════════════════════════════════════════════════════════
# 1. Base Configuration
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class EnvConfig:
    batch_size: int = 8
    max_problems: int = 2_000
    max_cases: int = 1_000_000
    split: str = "train"  # "train" | "valid" | "test"
    max_problem_length: int = 2_048
    max_solution_length: int = 512
    
    # Performance optimizations
    use_parallel_execution: bool = True    # Parallel code execution
    max_workers: int = None               # Auto-detect based on CPU count
    cache_extracted_functions: bool = True  # Cache function extraction
    precompute_test_hashes: bool = True   # Precompute test case hashes


class RewardType(Enum):
    SIMPLE = "simple"
    ERROR_TYPE = "error_type"


class EnvType(Enum):
    SIMPLE = "simple"
    CURRICULUM = "curriculum"
    MIXED_CURRICULUM = "mixed_curriculum"  # HumanEval + CodeContest


# ═══════════════════════════════════════════════════════════════════════════
# 2. Dataset & Execution
# ═══════════════════════════════════════════════════════════════════════════

# Simple LRU cache for function extraction
class SimpleLRUCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)


class CodeContestDataset:
    """DeepMind CodeContests dataset loader"""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.tasks: List[Dict[str, Any]] = []
        self._test_hashes = {}  # Cache for test case hashes
        self._load_dataset()

    def _load_dataset(self):
        logger.info(f"🔄 Loading {self.cfg.split} CodeContest dataset with optimizations...")
        
        ds = load_dataset(
            "deepmind/code_contests",
            split=self.cfg.split,
            streaming=False,
        )
        
        # Vectorized filtering for better performance
        ds = ds.filter(self._is_valid_task, batched=False, desc="Filtering tasks")
        ds = ds.filter(self._has_good_length, batched=False, desc="Filtering by length")
        
        num_problems = min(self.cfg.max_problems, len(ds))
        ds = ds.select(range(num_problems))
        
        # Process tasks with progress bar
        for row in tqdm(ds, desc=f"Processing {self.cfg.split} tasks"):
            task = self._to_task(row)
            if task:  # Only add valid tasks
                self.tasks.append(task)
        
        random.shuffle(self.tasks)
        logger.info(f"✅ Loaded {len(self.tasks)} CodeContest tasks")

    def _is_valid_task(self, row) -> bool:
        """Fast validation check"""
        return (row.get("time_limit") is not None and 
                row.get("private_tests") is not None and 
                row.get("description") is not None and 
                row.get("cf_tags") is not None and
                len(row.get("private_tests", {}).get("input", [])) > 0)

    def _has_good_length(self, row) -> bool:
        """Check description length"""
        desc = row.get("description", "")
        return 50 <= len(desc.strip()) <= self.cfg.max_problem_length

    def _to_task(self, row) -> Optional[Dict[str, Any]]:
        """Convert row to task with error handling"""
        try:
            tlim = row["time_limit"]
            if isinstance(tlim, dict):
                if "seconds" in tlim:
                    tlim = tlim["seconds"]
                else:
                    tlim = tlim.get("nanos", 1e9) / 1e9
                    

            tests = list(
                zip(
                    row["private_tests"]["input"][: self.cfg.max_cases],
                    row["private_tests"]["output"][: self.cfg.max_cases],
                )
            )
            
            task = {
                "name": row["name"],
                "description": row["description"],
                "tests": tests,
                "time_limit": float(tlim),
                "cf_tags": row.get("cf_tags", []),
                "dataset_type": "codecontest"
            }
            
            # Precompute test hashes if enabled
            if self.cfg.precompute_test_hashes:
                task_hash = self._compute_task_hash(task)
                self._test_hashes[task_hash] = [
                    hashlib.md5(f"{inp}{out}".encode()).hexdigest()
                    for inp, out in tests
                ]
                task["_hash"] = task_hash
            
            return task
        except Exception as e:
            logging.error(f"⚠️  Error processing task: {e}")
            return None

    def _compute_task_hash(self, task: Dict[str, Any]) -> str:
        """Compute hash for task identification"""
        return hashlib.md5(task["name"].encode()).hexdigest()

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks


class HumanEvalDataset:
    """HumanEval dataset loader for easier problems"""
    
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.tasks: List[Dict[str, Any]] = []
        self._load_dataset()
    
    def _load_dataset(self):
        logger.info(f"🔄 Loading HumanEval dataset...")
        
        try:
            ds = load_dataset("openai_humaneval")["test"]
            
            for row in ds:
                task = {
                    "name": f"humaneval_{row['task_id']}",
                    "description": row["prompt"],  # 함수 시그니처 + docstring
                    "canonical_solution": row["canonical_solution"],
                    "test": row["test"],  # assert statements
                    "entry_point": row["entry_point"],  # function name
                    "time_limit": 2.0,  # Default timeout
                    "cf_tags": ["implementation"],  # Mark as easy for curriculum
                    "dataset_type": "humaneval"
                }
                self.tasks.append(task)
            
            logger.info(f"✅ Loaded {len(self.tasks)} HumanEval tasks")
            
        except Exception as e:
            logger.error(f"⚠️  Failed to load HumanEval: {e}")
            self.tasks = []
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks


class MixedDatasetLoader:
    """Combines HumanEval and CodeContest datasets"""
    
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.humaneval_dataset = HumanEvalDataset(cfg)
        self.codecontest_dataset = CodeContestDataset(cfg)
        self.tasks: List[Dict[str, Any]] = []
        self._combine_datasets()
    
    def _combine_datasets(self):
        """Combine both datasets with priority to HumanEval for easier start"""
        
        # Get HumanEval tasks (mark as easy)
        humaneval_tasks = self.humaneval_dataset.get_all_tasks()
        for task in humaneval_tasks:
            task["difficulty_score"] = 1  # Easiest
            task["cf_tags"] = ["implementation", "easy"]  # Force easy classification
        
        # Get CodeContest tasks  
        codecontest_tasks = self.codecontest_dataset.get_all_tasks()
        for task in codecontest_tasks:
            task["difficulty_score"] = 3  # Harder
            if "dataset_type" not in task:
                task["dataset_type"] = "codecontest"
        
        # Combine with HumanEval first for curriculum
        self.tasks = humaneval_tasks + codecontest_tasks
        
        logger.info(f"✅ Combined datasets: {len(humaneval_tasks)} HumanEval + {len(codecontest_tasks)} CodeContest = {len(self.tasks)} total")
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks


class SafeCodeExecutor:
    """Safe code execution with caching and parallel processing"""
    
    # Class-level cache for extracted functions
    _function_cache = SimpleLRUCache(max_size=2000)
    
    @staticmethod
    def extract_solve_function(response: str) -> Optional[str]:
        """Function extraction with caching"""
        if SafeCodeExecutor._function_cache:
            # Create cache key
            cache_key = hashlib.md5(response.encode()).hexdigest()
            cached_result = SafeCodeExecutor._function_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Extract function
        result = SafeCodeExecutor._extract_function_impl(response)
        
        # Cache result
        if SafeCodeExecutor._function_cache:
            SafeCodeExecutor._function_cache.put(cache_key, result)
        
        return result
    
    @staticmethod
    def _extract_function_impl(response: str) -> Optional[str]:
        """Implementation of function extraction"""
        # 1. 코드 블록에서 solve 함수 찾기
        block_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", response)
        if block_match:
            src = block_match.group(1)
            solve_match = re.search(r"(def +solve\(.*?\)[\s\S]+?)(?:\n(?=def|\w)|$)", src)
            if solve_match:
                return solve_match.group(1).strip()
        
        # 2. 원본 텍스트에서 직접 solve 함수 찾기
        solve_match = re.search(r"(def +solve\(.*?\)[\s\S]+?)(?:\n(?=def|\w)|$)", response)
        if solve_match:
            return solve_match.group(1).strip()
        
        # 3. 더 관대한 패턴
        flexible_match = re.search(r"(def +\w*solve\w*\(.*?\)[\s\S]+?)(?:\n(?=def|\w)|$)", response)
        if flexible_match:
            return flexible_match.group(1).strip()
        
        return None

    @staticmethod
    def _run(code_str: str, input_str: str, q, timeout: float):
        """Single execution worker - 개선된 timeout 처리"""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(
                "import sys\n"
                f"{code_str}\n"
                "if __name__ == '__main__':\n"
                "    data = sys.stdin.read()\n"
                "    print(str(solve(data)).strip())\n"
            )
            path = f.name
        
        try:
            proc = subprocess.Popen(
                [sys.executable, path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # 프로세스 그룹 생성으로 더 확실한 종료
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            
            try:
                out, err = proc.communicate(input=input_str, timeout=timeout)
                status = "success" if proc.returncode == 0 else "error"
                q.put((status, out.strip() if status == "success" else err.strip()))
            except subprocess.TimeoutExpired:
                # 더 확실한 프로세스 종료
                try:
                    if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                        # 프로세스 그룹 전체 종료
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        proc.terminate()
                    
                    # 잠시 기다렸다가 강제 종료
                    try:
                        proc.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        else:
                            proc.kill()
                            
                except (ProcessLookupError, OSError):
                    # 프로세스가 이미 종료된 경우
                    pass
                
                q.put(("timeout", "TimeoutError"))
                
        except Exception as e:
            q.put(("error", str(e)))
        finally:
            try:
                os.unlink(path)
            except:
                pass

    @staticmethod
    def execute_code_safely(code: str, inp: str, timeout: float) -> Tuple[bool, str]:
        """Execute code safely with proper timeout handling"""
        
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=SafeCodeExecutor._run, 
            args=(code, inp, q, timeout)
        )
        p.start()
        p.join(timeout + 0.5)  # Give small buffer
        
        ok, out = False, ""
        if not q.empty():
            status, out = q.get()
            ok = status == "success"
        else:
            # Process didn't finish in time
            out = "TimeoutError"
        
        # Force cleanup
        if p.is_alive():
            p.terminate()
            p.join(0.1)
        if p.is_alive():
            p.kill()
            
        return ok, out

    @staticmethod
    def execute_multiple_parallel(tasks: List[Tuple[str, str, float]], max_workers: int = None) -> List[Tuple[bool, str]]:
        """Execute multiple code tasks in parallel"""
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count() // 2, len(tasks))
        
        if max_workers == 1 or len(tasks) <= 1:
            # Single-threaded fallback
            return [
                SafeCodeExecutor.execute_code_safely(code, inp, timeout)
                for code, inp, timeout in tasks
            ]
        
        # Parallel execution
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(SafeCodeExecutor.execute_code_safely, code, inp, timeout)
                    for code, inp, timeout in tasks
                ]
                results = [future.result() for future in futures]
            return results
        except Exception as e:
            logger.warning(f"⚠️  Parallel execution failed: {e}, falling back to sequential")
            return [
                SafeCodeExecutor.execute_code_safely(code, inp, timeout)
                for code, inp, timeout in tasks
            ]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Reward Calculators
# ═══════════════════════════════════════════════════════════════════════════
class BaseRewardCalculator(ABC):
    """Base reward calculator interface"""
    
    @abstractmethod
    def calculate_reward(self, task: Dict[str, Any], response: str) -> float:
        pass


class SimpleRewardCalculator(BaseRewardCalculator):
    """Simple binary/fractional reward with parallel execution"""
    
    def __init__(self, use_parallel: bool = True, max_workers: int = None):
        self.use_parallel = use_parallel
        self.max_workers = max_workers
    
    def calculate_reward(self, task: Dict[str, Any], response: str) -> float:
        code = SafeCodeExecutor.extract_solve_function(response)
        if not code:
            return -1.0  # Parse error

        tests = task["tests"]
        if not tests:
            return -1.0
        
        # Apply task-specific timeout with reasonable limits
        timeout = task.get("time_limit", 2.0)
        timeout = min(max(timeout, 0.5), 3.0)  # Clamp between 0.5-3 seconds
        
        # Limit number of test cases for faster training
        max_tests = min(len(tests), 5)  # Max 5 test cases per problem
        tests = tests[:max_tests]
        
        if self.use_parallel and len(tests) > 1:
            # Parallel execution for multiple test cases
            execution_tasks = [
                (code, inp, timeout) for inp, expected_out in tests
            ]
            
            results = SafeCodeExecutor.execute_multiple_parallel(
                execution_tasks, self.max_workers
            )
            
            passed = sum(
                1 for (ok, result), (_, expected_out) in zip(results, tests)
                if ok and result.strip() == expected_out.strip()
            )
        else:
            # Sequential execution
            passed = 0
            for inp, expected_out in tests:
                ok, result = SafeCodeExecutor.execute_code_safely(
                    code, inp, timeout
                )
                if ok and result.strip() == expected_out.strip():
                    passed += 1
        
        return passed / len(tests)


class HumanEvalRewardCalculator(BaseRewardCalculator):
    """Dense reward calculator for HumanEval problems"""
    
    def __init__(self, use_parallel: bool = True, max_workers: int = None):
        self.use_parallel = use_parallel
        self.max_workers = max_workers
    
    def calculate_reward(self, task: Dict[str, Any], response: str) -> float:
        """Calculate dense reward for HumanEval tasks"""
        
        if task.get("dataset_type") != "humaneval":
            # Fallback to standard reward calculation for non-HumanEval tasks
            return self._calculate_codecontest_reward(task, response)
        
        reward = 0.0
        
        # 1. Syntax check (0.2 points)
        try:
            ast.parse(response)
            reward += 0.2
        except SyntaxError:
            return -0.3  # Syntax error penalty
        
        # 2. Function definition check (0.2 points)
        entry_point = task.get("entry_point", "")
        if entry_point and f"def {entry_point}(" in response:
            reward += 0.2
        elif "def " in response:  # Any function definition
            reward += 0.1
        
        # 3. Basic execution test (0.2 points)
        try:
            exec(response)
            reward += 0.2
        except Exception:
            return reward - 0.1  # Execution error but syntax OK
        
        # 4. Test execution (0.4 points)
        try:
            # Combine function code with test assertions
            test_code = task.get("test", "")
            if test_code:
                full_code = response + "\n" + test_code
                exec(full_code)
                reward += 0.4  # All tests passed
            else:
                reward += 0.2  # No tests to run, partial credit
                
        except AssertionError:
            # Function runs but tests fail
            reward += 0.1
        except Exception:
            # Runtime error in tests
            pass
        
        return min(reward, 1.0)  # Cap at 1.0
    
    def _calculate_codecontest_reward(self, task: Dict[str, Any], response: str) -> float:
        """Fallback to CodeContest-style reward calculation"""
        # Use existing SimpleRewardCalculator logic
        simple_calc = SimpleRewardCalculator(self.use_parallel, self.max_workers)
        return simple_calc.calculate_reward(task, response)


class ErrorTypeRewardCalculator(BaseRewardCalculator):
    """Error-type aware reward with parallel execution"""
    
    def __init__(self, use_parallel: bool = True, max_workers: int = None):
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        
        # Reward parameters
        self.perfect_reward = 1.0
        self.partial_min = 0.1
        self.partial_max = 0.7
        self.syntax_error_penalty = -0.3
        self.runtime_error_penalty = -0.1
        self.timeout_penalty = -0.2
        self.parse_error_penalty = -0.5
    
    def calculate_reward(self, task: Dict[str, Any], response: str) -> float:
        code = SafeCodeExecutor.extract_solve_function(response)
        if not code:
            return self.parse_error_penalty
        
        execution_results = self._analyze_execution(task, code)
        
        passed = execution_results["passed"]
        total = execution_results["total"]
        syntax_errors = execution_results["syntax_errors"]
        runtime_errors = execution_results["runtime_errors"]
        timeouts = execution_results["timeouts"]
        
        if total == 0:
            return self.parse_error_penalty
        
        pass_ratio = passed / total
        
        # Perfect solution
        if pass_ratio == 1.0:
            return self.perfect_reward
        
        # Partial solutions with pass ratio
        elif pass_ratio > 0:
            partial_reward = (self.partial_min + 
                            (self.partial_max - self.partial_min) * pass_ratio)
            return partial_reward
        
        # No passes - analyze error types
        elif timeouts > 0:
            return self.timeout_penalty
        elif syntax_errors > 0:
            return self.syntax_error_penalty
        elif runtime_errors > 0:
            return self.runtime_error_penalty
        else:
            # Logic error (runs but wrong output)
            return 0.0
    
    def _analyze_execution(self, task: Dict[str, Any], code: str) -> Dict[str, Any]:
        """Execution analysis with parallel processing"""
        tests = task["tests"]
        # Apply task-specific timeout with reasonable limits
        timeout = task.get("time_limit", 2.0)
        timeout = min(max(timeout, 0.5), 3.0)  # Clamp between 0.5-3 seconds
        
        # Limit test cases for faster training
        max_tests = min(len(tests), 5)  # Max 5 test cases per problem
        tests = tests[:max_tests]
        
        if self.use_parallel and len(tests) > 1:
            # Parallel execution
            execution_tasks = [
                (code, inp, timeout) for inp, expected_out in tests
            ]
            
            results = SafeCodeExecutor.execute_multiple_parallel(
                execution_tasks, self.max_workers
            )
            
            # Analyze results
            passed = 0
            syntax_errors = 0
            runtime_errors = 0
            timeouts = 0
            
            for (ok, result), (_, expected_out) in zip(results, tests):
                if ok and result.strip() == expected_out.strip():
                    passed += 1
                elif not ok:
                    # Classify error types
                    if "TimeoutError" in result:
                        timeouts += 1
                    elif any(err in result for err in ["SyntaxError", "IndentationError", "TabError"]):
                        syntax_errors += 1
                    else:
                        runtime_errors += 1
        else:
            # Sequential execution
            passed = 0
            syntax_errors = 0
            runtime_errors = 0
            timeouts = 0
            
            for inp, expected_out in tests:
                ok, result = SafeCodeExecutor.execute_code_safely(
                    code, inp, timeout
                )
                
                if ok and result.strip() == expected_out.strip():
                    passed += 1
                elif not ok:
                    # Classify error types
                    if "TimeoutError" in result:
                        timeouts += 1
                    elif any(err in result for err in ["SyntaxError", "IndentationError", "TabError"]):
                        syntax_errors += 1
                    else:
                        runtime_errors += 1
        
        return {
            "passed": passed,
            "total": len(tests),
            "syntax_errors": syntax_errors,
            "runtime_errors": runtime_errors,
            "timeouts": timeouts
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Curriculum Learning Components
# ═══════════════════════════════════════════════════════════════════════════
class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class CurriculumConfig:
    # CF Tags 기반 난이도 분류
    easy_tags: Set[str] = None
    medium_tags: Set[str] = None  
    hard_tags: Set[str] = None
    
    # Performance thresholds
    pass_rate_threshold: float = 0.05
    avg_reward_threshold: float = 0.0
    min_episodes_per_level: int = 50
    eval_window_size: int = 20
    
    # Performance optimization
    cache_difficulty_classification: bool = True
    
    def __post_init__(self):
        if self.easy_tags is None:
            self.easy_tags = {
                "implementation", "math", "brute force", "constructive algorithms",
                "greedy", "sortings", "strings", "binary search", "two pointers",
            }
        
        if self.medium_tags is None:
            self.medium_tags = {
                "data structures", "dfs and similar", "graphs", "trees", "dp",
                "number theory", "combinatorics", "bitmasks", "hashing",
                "divide and conquer", "ternary search",
            }
            
        if self.hard_tags is None:
            self.hard_tags = {
                "shortest paths", "flows", "graph matchings", "dsu",
                "string suffix structures", "matrices", "geometry", "fft",
                "chinese remainder theorem", "meet-in-the-middle",
                "expression parsing", "probabilities", "games", "interactive", "*special"
            }


class CurriculumManager:
    """Curriculum learning logic manager with caching"""
    
    def __init__(self, problems: List[Dict[str, Any]], config: CurriculumConfig):
        self.config = config
        self.current_level = DifficultyLevel.EASY
        self.episodes_at_current_level = 0
        self.performance_history: List[Dict[str, float]] = []
        
        # Cache for difficulty classifications
        self._difficulty_cache = {}
        
        # Categorize problems by difficulty
        self.problems_by_difficulty = self._categorize_problems(problems)
        
        logger.info((
            f"📚 Curriculum Learning initialized:\n"
            f"  Easy: {len(self.problems_by_difficulty[DifficultyLevel.EASY])}\n"
            f"  Medium: {len(self.problems_by_difficulty[DifficultyLevel.MEDIUM])}\n"
            f"  Hard: {len(self.problems_by_difficulty[DifficultyLevel.HARD])}\n"
            f"  Starting with: {self.current_level.value}"
        ))
        
        
    
    def _categorize_problems(self, problems: List[Dict[str, Any]]) -> Dict[DifficultyLevel, List[Dict[str, Any]]]:
        """Problem categorization with caching"""
        categories = {
            DifficultyLevel.EASY: [],
            DifficultyLevel.MEDIUM: [],
            DifficultyLevel.HARD: []
        }
        
        for problem in tqdm(problems, desc="Categorizing problems"):
            problem_id = problem.get("name", str(hash(str(problem))))
            
            # Check cache first
            if self.config.cache_difficulty_classification and problem_id in self._difficulty_cache:
                difficulty = self._difficulty_cache[problem_id]
            else:
                cf_tags = self._extract_tags(problem)
                difficulty = self._classify_by_tags(cf_tags)
                
                # Cache result
                if self.config.cache_difficulty_classification:
                    self._difficulty_cache[problem_id] = difficulty
            
            categories[difficulty].append(problem)
        
        # Rebalance if needed
        self._rebalance_categories(categories)
        return categories
    
    def _extract_tags(self, problem: Dict[str, Any]) -> List[str]:
        """Extract tags"""
        possible_fields = ["cf_tags", "tags", "cf_tag", "tag"]
        
        for field in possible_fields:
            if field in problem and problem[field]:
                tags = problem[field]
                if isinstance(tags, list):
                    return [str(tag).lower().strip() for tag in tags]
                elif isinstance(tags, str):
                    return [tag.strip().lower() for tag in tags.replace(',', ' ').split()]
        return []
    
    def _classify_by_tags(self, tags: List[str]) -> DifficultyLevel:
        """Classify by tags"""
        tags_set = set(tags)
        
        if tags_set.intersection(self.config.hard_tags):
            return DifficultyLevel.HARD
        elif tags_set.intersection(self.config.medium_tags):
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY
    
    def _rebalance_categories(self, categories: Dict[DifficultyLevel, List[Dict[str, Any]]]):
        """Ensure balanced distribution"""
        easy_count = len(categories[DifficultyLevel.EASY])
        medium_count = len(categories[DifficultyLevel.MEDIUM]) 
        hard_count = len(categories[DifficultyLevel.HARD])
        total = easy_count + medium_count + hard_count
        
        if total == 0:
            return
        
        # If Easy is too small (< 10%), move some from Medium
        if easy_count / total < 0.1 and medium_count > 50:
            move_count = min(50, medium_count // 3)
            moved_problems = categories[DifficultyLevel.MEDIUM][:move_count]
            categories[DifficultyLevel.EASY].extend(moved_problems)
            categories[DifficultyLevel.MEDIUM] = categories[DifficultyLevel.MEDIUM][move_count:]
        
        # If Medium is too small (< 20%), move some from Hard
        if medium_count / total < 0.2 and hard_count > 30:
            move_count = min(30, hard_count // 4)
            moved_problems = categories[DifficultyLevel.HARD][:move_count]
            categories[DifficultyLevel.MEDIUM].extend(moved_problems)
            categories[DifficultyLevel.HARD] = categories[DifficultyLevel.HARD][move_count:]
    
    def get_current_problems(self) -> List[Dict[str, Any]]:
        """Get problems for current difficulty level"""
        return self.problems_by_difficulty[self.current_level]
    
    def record_performance(self, avg_reward: float, pass_rate: float):
        """Record performance and check for advancement"""
        self.episodes_at_current_level += 1
        self.performance_history.append({
            'level': self.current_level.value,
            'episode': self.episodes_at_current_level,
            'avg_reward': avg_reward,
            'pass_rate': pass_rate
        })
        
        if self._should_advance():
            self._advance_level()
    
    def _should_advance(self) -> bool:
        """Check if should advance"""
        if self.episodes_at_current_level < self.config.min_episodes_per_level:
            return False
        
        if self.current_level == DifficultyLevel.HARD:
            return False
        
        recent_episodes = [ep for ep in self.performance_history 
                          if ep['level'] == self.current_level.value][-self.config.eval_window_size:]
        
        if len(recent_episodes) < self.config.eval_window_size:
            return False
        
        avg_pass_rate = sum(ep['pass_rate'] for ep in recent_episodes) / len(recent_episodes)
        avg_reward = sum(ep['avg_reward'] for ep in recent_episodes) / len(recent_episodes)
        
        return (avg_pass_rate >= self.config.pass_rate_threshold and 
                avg_reward >= self.config.avg_reward_threshold)
    
    def _advance_level(self):
        """Advance to next level"""
        old_level = self.current_level
        
        if self.current_level == DifficultyLevel.EASY:
            self.current_level = DifficultyLevel.MEDIUM
        elif self.current_level == DifficultyLevel.MEDIUM:
            self.current_level = DifficultyLevel.HARD
        
        self.episodes_at_current_level = 0
        
        logger.info((
            f"🏆 CURRICULUM ADVANCED! 🏆\n"
            f"Progression: {old_level.value} → {self.current_level.value}\n"
            f"New challenge set: {len(self.problems_by_difficulty[self.current_level])} problems"
        ))

    
    def get_status(self) -> Dict[str, Any]:
        """Get status"""
        recent_episodes = [ep for ep in self.performance_history 
                          if ep['level'] == self.current_level.value][-self.config.eval_window_size:]
        
        if recent_episodes:
            recent_pass_rate = sum(ep['pass_rate'] for ep in recent_episodes) / len(recent_episodes)
            recent_avg_reward = sum(ep['avg_reward'] for ep in recent_episodes) / len(recent_episodes)
        else:
            recent_pass_rate = 0.0
            recent_avg_reward = -1.0
        
        return {
            'curriculum_level': self.current_level.value,
            'episodes_at_level': self.episodes_at_current_level,
            'recent_pass_rate': recent_pass_rate,
            'recent_avg_reward': recent_avg_reward,
            'problems_at_level': len(self.problems_by_difficulty[self.current_level]),
            'ready_to_advance': self._should_advance()
        }


class MixedCurriculumManager:
    """Curriculum manager that prioritizes HumanEval then moves to CodeContest"""
    
    def __init__(self, problems: List[Dict[str, Any]], config: CurriculumConfig):
        self.config = config
        self.current_level = DifficultyLevel.EASY
        self.episodes_at_current_level = 0
        self.performance_history: List[Dict[str, float]] = []
        
        # Separate problems by dataset type and difficulty
        self.problems_by_stage = self._categorize_problems_by_stage(problems)
        
        logger.info((
            f"📚 Mixed Curriculum Learning initialized:\n"
            f"  Stage 1 (HumanEval): {len(self.problems_by_stage['humaneval'])}\n"
            f"  Stage 2 (CodeContest Easy): {len(self.problems_by_stage['codecontest_easy'])}\n"
            f"  Stage 3 (CodeContest Medium): {len(self.problems_by_stage['codecontest_medium'])}\n"
            f"  Stage 4 (CodeContest Hard): {len(self.problems_by_stage['codecontest_hard'])}\n"
            f"  Starting with: HumanEval problems"
        ))
    
    def _categorize_problems_by_stage(self, problems: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize problems into curriculum stages"""
        
        stages = {
            'humaneval': [],
            'codecontest_easy': [],
            'codecontest_medium': [],
            'codecontest_hard': []
        }
        
        for problem in problems:
            dataset_type = problem.get("dataset_type", "codecontest")
            
            if dataset_type == "humaneval":
                stages['humaneval'].append(problem)
            else:
                # CodeContest problems - classify by tags
                cf_tags = self._extract_tags(problem)
                difficulty = self._classify_by_tags(cf_tags)
                
                if difficulty == DifficultyLevel.EASY:
                    stages['codecontest_easy'].append(problem)
                elif difficulty == DifficultyLevel.MEDIUM:
                    stages['codecontest_medium'].append(problem)
                else:
                    stages['codecontest_hard'].append(problem)
        
        return stages
    
    def _extract_tags(self, problem: Dict[str, Any]) -> List[str]:
        """Extract tags from problem"""
        possible_fields = ["cf_tags", "tags", "cf_tag", "tag"]
        
        for field in possible_fields:
            if field in problem and problem[field]:
                tags = problem[field]
                if isinstance(tags, list):
                    return [str(tag).lower().strip() for tag in tags]
                elif isinstance(tags, str):
                    return [tag.strip().lower() for tag in tags.replace(',', ' ').split()]
        return []
    
    def _classify_by_tags(self, tags: List[str]) -> DifficultyLevel:
        """Classify CodeContest problems by tags"""
        tags_set = set(tags)
        
        if tags_set.intersection(self.config.hard_tags):
            return DifficultyLevel.HARD
        elif tags_set.intersection(self.config.medium_tags):
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY
    
    def get_current_problems(self) -> List[Dict[str, Any]]:
        """Get problems for current stage"""
        
        # Stage progression: HumanEval -> CodeContest Easy -> Medium -> Hard
        if self.episodes_at_current_level < 30:  # First 30 episodes: HumanEval only
            current_problems = self.problems_by_stage['humaneval']
            
        elif self.episodes_at_current_level < 80:  # Episodes 30-80: Mix HumanEval + Easy CodeContest
            humaneval_problems = self.problems_by_stage['humaneval']
            easy_problems = self.problems_by_stage['codecontest_easy']
            current_problems = humaneval_problems + easy_problems
            
        elif self.episodes_at_current_level < 150:  # Episodes 80-150: Easy + Medium CodeContest
            easy_problems = self.problems_by_stage['codecontest_easy']
            medium_problems = self.problems_by_stage['codecontest_medium']
            current_problems = easy_problems + medium_problems
            
        else:  # Episodes 150+: All problems
            current_problems = (
                self.problems_by_stage['codecontest_easy'] + 
                self.problems_by_stage['codecontest_medium'] + 
                self.problems_by_stage['codecontest_hard']
            )
        
        return current_problems if current_problems else self.problems_by_stage['humaneval']
    
    def record_performance(self, avg_reward: float, pass_rate: float):
        """Record performance"""
        self.episodes_at_current_level += 1
        self.performance_history.append({
            'level': self._get_current_stage_name(),
            'episode': self.episodes_at_current_level,
            'avg_reward': avg_reward,
            'pass_rate': pass_rate
        })
    
    def _get_current_stage_name(self) -> str:
        """Get current stage name for logging"""
        if self.episodes_at_current_level < 30:
            return "humaneval_only"
        elif self.episodes_at_current_level < 80:
            return "humaneval_mixed"
        elif self.episodes_at_current_level < 150:
            return "codecontest_easy_medium"
        else:
            return "codecontest_all"
    
    def get_status(self) -> Dict[str, Any]:
        """Get curriculum status"""
        recent_episodes = self.performance_history[-20:] if len(self.performance_history) >= 20 else self.performance_history
        
        if recent_episodes:
            recent_pass_rate = sum(ep['pass_rate'] for ep in recent_episodes) / len(recent_episodes)
            recent_avg_reward = sum(ep['avg_reward'] for ep in recent_episodes) / len(recent_episodes)
        else:
            recent_pass_rate = 0.0
            recent_avg_reward = -1.0
        
        current_problems = self.get_current_problems()
        
        return {
            'curriculum_level': self._get_current_stage_name(),
            'episodes_at_level': self.episodes_at_current_level,
            'recent_pass_rate': recent_pass_rate,
            'recent_avg_reward': recent_avg_reward,
            'problems_at_level': len(current_problems),
            'stage_description': self._get_stage_description()
        }
    
    def _get_stage_description(self) -> str:
        """Get description of current stage"""
        stage_name = self._get_current_stage_name()
        
        descriptions = {
            "humaneval_only": "Learning basics with HumanEval function completion",
            "humaneval_mixed": "Mixing HumanEval with easy CodeContest problems", 
            "codecontest_easy_medium": "Progressing through CodeContest easy/medium problems",
            "codecontest_all": "Tackling full range of CodeContest problems"
        }
        
        return descriptions.get(stage_name, "Unknown stage")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Environment Classes
# ═══════════════════════════════════════════════════════════════════════════
class BaseCodeEnv:
    """Base environment class"""
    
    def __init__(self, cfg: EnvConfig, reward_calculator: BaseRewardCalculator):
        self.cfg = cfg
        self.reward_calculator = reward_calculator
        self.dataset = CodeContestDataset(cfg).get_all_tasks()
        self.batch: List[Dict[str, Any]] = []
        self.performance_history: List[float] = []

    def reset_batch(self) -> List[Dict[str, Any]]:
        """Sample a new batch - implemented by subclasses"""
        raise NotImplementedError
    
    def step_batch(self, solutions: List[str], max_workers: Optional[int] = None) -> List[float]:
        """Batch execution with parallel processing"""
        if max_workers is None:
            if self.cfg.max_workers is not None:
                max_workers = self.cfg.max_workers
            else:
                max_workers = min(multiprocessing.cpu_count() // 2, len(solutions)) if self.cfg.use_parallel_execution else 1
        
        if max_workers == 1 or len(solutions) <= 1 or not self.cfg.use_parallel_execution:
            # Sequential execution
            rewards = [
                self.reward_calculator.calculate_reward(task, sol)
                for task, sol in tqdm(list(zip(self.batch, solutions)), desc="Evaluating Batch", leave=False)
            ]
        else:
            # Parallel execution using ThreadPoolExecutor for I/O bound reward calculation
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(self.reward_calculator.calculate_reward, task, sol)
                        for task, sol in zip(self.batch, solutions)
                    ]
                    rewards = [future.result() for future in tqdm(futures, desc="Evaluating Batch", leave=False)]
            except Exception as e:
                logger.warning(f"⚠️  Parallel reward calculation failed: {e}, falling back to sequential")
                rewards = [
                    self.reward_calculator.calculate_reward(task, sol)
                    for task, sol in tqdm(list(zip(self.batch, solutions)), desc="Evaluating Batch", leave=False)
                ]
        
        self.performance_history.extend(rewards)
        return rewards
    
    def get_all_problems(self) -> List[Dict[str, Any]]:
        return self.dataset
    
    def get_performance_stats(self, n_hist: int = 100) -> Dict[str, float]:
        """Get recent performance statistics"""
        if not self.performance_history:
            return {}
        
        recent = self.performance_history[-n_hist:] if n_hist > 0 else self.performance_history
        return {
            "avg_reward": sum(recent) / len(recent),
            "pass_rate": sum(1 for r in recent if r >= 0.99) / len(recent),
            "positive_rate": sum(1 for r in recent if r > 0) / len(recent),
        }


class SimpleCodeEnv(BaseCodeEnv):
    """Simple environment - random sampling from all problems"""
    
    def __init__(self, cfg: EnvConfig, reward_calculator: BaseRewardCalculator):
        super().__init__(cfg, reward_calculator)
        self.reset_batch()
    
    def reset_batch(self) -> List[Dict[str, Any]]:
        self.batch = random.sample(
            self.dataset, 
            min(self.cfg.batch_size, len(self.dataset))
        )
        return self.batch


class CurriculumCodeEnv(BaseCodeEnv):
    """Curriculum learning environment"""
    
    def __init__(self, cfg: EnvConfig, reward_calculator: BaseRewardCalculator, 
                 curriculum_config: CurriculumConfig = None):
        # STEP 1: 부모 클래스 초기화
        super().__init__(cfg, reward_calculator)
        
        # STEP 2: Curriculum manager 설정
        if curriculum_config is None:
            curriculum_config = CurriculumConfig()
        
        self.curriculum_manager = CurriculumManager(self.dataset, curriculum_config)
        
        # STEP 3: 이제 reset_batch 호출
        self.reset_batch()
    
    def reset_batch(self) -> List[Dict[str, Any]]:
        # curriculum_manager가 None이 아닌지 확인
        if not hasattr(self, 'curriculum_manager') or self.curriculum_manager is None:
            # fallback: 모든 문제에서 랜덤 샘플링
            self.batch = random.sample(
                self.dataset, 
                min(self.cfg.batch_size, len(self.dataset))
            )
            return self.batch
        
        current_problems = self.curriculum_manager.get_current_problems()
        
        if not current_problems:
            logger.warning(f"⚠️  No problems available for current level, using all problems")
            current_problems = self.dataset
        
        batch_size = min(self.cfg.batch_size, len(current_problems))
        self.batch = random.sample(current_problems, batch_size)
        return self.batch
    
    def record_episode_performance(self, avg_reward: float, pass_rate: float):
        """Record performance for curriculum advancement"""
        if hasattr(self, 'curriculum_manager') and self.curriculum_manager is not None:
            self.curriculum_manager.record_performance(avg_reward, pass_rate)
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get curriculum status"""
        if hasattr(self, 'curriculum_manager') and self.curriculum_manager is not None:
            return self.curriculum_manager.get_status()
        else:
            return {
                'curriculum_level': 'unknown',
                'episodes_at_level': 0,
                'recent_pass_rate': 0.0,
                'recent_avg_reward': -1.0,
                'problems_at_level': len(self.dataset),
                'ready_to_advance': False
            }


class MixedCurriculumCodeEnv(BaseCodeEnv):
    """Environment that uses both HumanEval and CodeContest with curriculum"""
    
    def __init__(self, cfg: EnvConfig, reward_calculator: BaseRewardCalculator, 
                 curriculum_config: CurriculumConfig = None):
        
        # Use mixed dataset instead of CodeContest only
        self.cfg = cfg
        self.reward_calculator = reward_calculator
        self.dataset = MixedDatasetLoader(cfg).get_all_tasks()  # Mixed dataset
        self.batch: List[Dict[str, Any]] = []
        self.performance_history: List[float] = []
        
        # Setup mixed curriculum manager
        if curriculum_config is None:
            curriculum_config = CurriculumConfig(
                # More relaxed thresholds for mixed curriculum
                pass_rate_threshold=0.02,  # 2% pass rate to advance
                avg_reward_threshold=-0.3,  # Average reward > -0.3
                min_episodes_per_level=20,
                eval_window_size=15
            )
        
        self.curriculum_manager = MixedCurriculumManager(self.dataset, curriculum_config)
        self.reset_batch()
    
    def reset_batch(self) -> List[Dict[str, Any]]:
        """Sample batch from current curriculum stage"""
        if not hasattr(self, 'curriculum_manager') or self.curriculum_manager is None:
            # Fallback: random sampling
            self.batch = random.sample(
                self.dataset, 
                min(self.cfg.batch_size, len(self.dataset))
            )
            return self.batch
        
        current_problems = self.curriculum_manager.get_current_problems()
        
        if not current_problems:
            logger.warning(f"⚠️  No problems available for current stage, using all problems")
            current_problems = self.dataset
        
        batch_size = min(self.cfg.batch_size, len(current_problems))
        self.batch = random.sample(current_problems, batch_size)
        return self.batch
    
    def record_episode_performance(self, avg_reward: float, pass_rate: float):
        """Record performance for curriculum advancement"""
        if hasattr(self, 'curriculum_manager') and self.curriculum_manager is not None:
            self.curriculum_manager.record_performance(avg_reward, pass_rate)
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get curriculum status"""
        if hasattr(self, 'curriculum_manager') and self.curriculum_manager is not None:
            return self.curriculum_manager.get_status()
        else:
            return {
                'curriculum_level': 'mixed_unknown',
                'episodes_at_level': 0,
                'recent_pass_rate': 0.0,
                'recent_avg_reward': -1.0,
                'problems_at_level': len(self.dataset),
                'stage_description': 'Mixed dataset without curriculum'
            }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Environment Factory
# ═══════════════════════════════════════════════════════════════════════════
class EnvironmentFactory:
    """Factory for creating environment combinations"""
    
    @staticmethod
    def create_environment(env_type: EnvType, reward_type: RewardType, 
                          cfg: EnvConfig, curriculum_config: CurriculumConfig = None):
        """Create environment with specified type and reward calculator"""
        
        # Create reward calculator
        if reward_type == RewardType.SIMPLE:
            if env_type == EnvType.MIXED_CURRICULUM:
                # Use HumanEval-aware reward calculator for mixed environment
                reward_calculator = HumanEvalRewardCalculator(
                    use_parallel=cfg.use_parallel_execution,
                    max_workers=cfg.max_workers
                )
            else:
                reward_calculator = SimpleRewardCalculator(
                    use_parallel=cfg.use_parallel_execution,
                    max_workers=cfg.max_workers
                )
        elif reward_type == RewardType.ERROR_TYPE:
            reward_calculator = ErrorTypeRewardCalculator(
                use_parallel=cfg.use_parallel_execution,
                max_workers=cfg.max_workers
            )
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        # Create environment
        if env_type == EnvType.SIMPLE:
            return SimpleCodeEnv(cfg, reward_calculator)
        elif env_type == EnvType.CURRICULUM:
            return CurriculumCodeEnv(cfg, reward_calculator, curriculum_config)
        elif env_type == EnvType.MIXED_CURRICULUM:
            return MixedCurriculumCodeEnv(cfg, reward_calculator, curriculum_config)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════
def create_env(env_name: str, cfg: EnvConfig, curriculum_config: CurriculumConfig = None):
    """
    Environment creation function with HumanEval + CodeContest mixed curriculum
    
    Available environments:
    - "simple_simple": Simple env + Simple reward
    - "simple_error": Simple env + Error-type reward
    - "curriculum_simple": Curriculum env + Simple reward  
    - "curriculum_error": Curriculum env + Error-type reward
    - "mixed_curriculum": HumanEval + CodeContest curriculum (NEW!)
    """
    
    env_mapping = {
        "simple_simple": (EnvType.SIMPLE, RewardType.SIMPLE),
        "simple_error": (EnvType.SIMPLE, RewardType.ERROR_TYPE),
        "curriculum_simple": (EnvType.CURRICULUM, RewardType.SIMPLE),
        "curriculum_error": (EnvType.CURRICULUM, RewardType.ERROR_TYPE),
        "mixed_curriculum": (EnvType.MIXED_CURRICULUM, RewardType.SIMPLE),
    }
    
    if env_name not in env_mapping:
        available = list(env_mapping.keys())
        raise ValueError(f"Unknown environment '{env_name}'. Available: {available}")
    
    env_type, reward_type = env_mapping[env_name]
    
    # Apply performance optimizations if not explicitly set
    if cfg.max_workers is None:
        cfg.max_workers = min(multiprocessing.cpu_count() // 2, 8)
    
    return EnvironmentFactory.create_environment(env_type, reward_type, cfg, curriculum_config)


def create_optimized_env_config(
    batch_size: int = 8,
    max_problems: int = 500,  # Reduced for faster loading
    split: str = "train",
    enable_parallel: bool = True,
    max_workers: int = None
) -> EnvConfig:
    """Create environment configuration with performance settings"""
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count() // 2, 4)  # More conservative
    
    return EnvConfig(
        batch_size=batch_size,
        max_problems=max_problems,
        max_cases=5,  # Limit test cases per problem
        split=split,
        use_parallel_execution=enable_parallel,
        max_workers=max_workers,
        cache_extracted_functions=True,
        precompute_test_hashes=False  # Skip for speed
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. Usage Example
# ═══════════════════════════════════════════════════════════════════════════
def example_usage():
    """Example of using the new mixed curriculum environment"""
    
    # Create config
    cfg = EnvConfig(
        batch_size=32,
        max_problems=1000,  # Reduced for faster loading
        split="train"
    )
    
    # Create mixed curriculum environment
    env = create_env("mixed_curriculum", cfg)
    
    # Check initial curriculum status
    status = env.get_curriculum_status()
    print(f"Starting stage: {status['stage_description']}")
    print(f"Problems available: {status['problems_at_level']}")
    
    # Simulate training loop
    for episode in range(10):
        # Reset and get problems
        problems = env.reset_batch()
        
        # Dummy solutions
        solutions = ["def solution(): return 42"] * len(problems)
        
        # Get rewards
        rewards = env.step_batch(solutions)
        
        # Record performance
        avg_reward = sum(rewards) / len(rewards)
        pass_rate = sum(1 for r in rewards if r >= 0.5) / len(rewards)
        env.record_episode_performance(avg_reward, pass_rate)
        
        print(f"Episode {episode}: Avg reward = {avg_reward:.3f}, Pass rate = {pass_rate:.3f}")
        
        # Check curriculum status every few episodes
        if episode % 5 == 0:
            status = env.get_curriculum_status()
            print(f"  Current stage: {status['stage_description']}")


if __name__ == "__main__":
    example_usage()