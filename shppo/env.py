"""
yy_env.py - Environment & Reward System
성능 최적화: 병렬 처리, 캐싱, 메모리 관리
"""

from __future__ import annotations
import os
import re
import sys
import tempfile
import subprocess
import multiprocessing
import random
import hashlib
import signal
import shutil
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, TypedDict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datasets import load_dataset

# ═══════════════════════════════════════════════════════════════════════════
# 1. Base Configuration
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class EnvConfig:
    batch_size: int = 8
    max_problems: int = 2_000
    max_cases: int = 1_000_000
    split: str = "train"  # "train" | "validation" | "test"
    max_problem_length: int = 2_048
    max_solution_length: int = 512
    use_parallel_execution: bool = True    # Parallel code execution
    max_workers: Optional[int] = None     # Auto-detect based on CPU count
    cache_extracted_functions: bool = True
    precompute_test_hashes: bool = True
    test_timeout: float = 5.0             # seconds per test
    use_temp_dir: bool = True             # Use system temp directory
    temp_dir_prefix: str = "shppo_env_"   # Prefix for temp directories
    cleanup_on_exit: bool = True          # Clean up temp dirs when done

class RewardType(Enum):
    SIMPLE = "simple"

class EnvType(Enum):
    SIMPLE = "simple"

# ═══════════════════════════════════════════════════════════════════════════
# 2. Simple LRU Cache for function extraction
# ═══════════════════════════════════════════════════════════════════════════
class SimpleLRUCache:
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, str] = {}
        self.order: List[str] = []
        self.max_size = max_size

    def get(self, key: str) -> Optional[str]:
        val = self.cache.get(key)
        if val is not None:
            self.order.remove(key)
            self.order.append(key)
        return val

    def put(self, key: str, value: str) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.order) >= self.max_size:
            old = self.order.pop(0)
            del self.cache[old]
        self.cache[key] = value
        self.order.append(key)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Dataset Loader using HuggingFace DeepMind CodeContests
# ═══════════════════════════════════════════════════════════════════════════
class CodeContestDataset:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.tasks: List[Dict[str, Any]] = []
        self._hashes: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        ds = load_dataset(
            "deepmind/code_contests", split=self.cfg.split, streaming=False
        )
        # filter invalid or too long
        def valid(example):
            return (
                example.get("description")
                and len(example["description"]) <= self.cfg.max_problem_length
                and example.get("private_tests")
                and len(example["private_tests"]["input"]) > 0
                and example.get("public_tests")
                and len(example["public_tests"]["input"]) > 0
            )
        ds = ds.filter(valid)
        size = min(self.cfg.max_problems, len(ds))
        ds = ds.select(range(size))

        for row in tqdm(ds, desc="Loading tasks"):
            try:
                priv = list(zip(
                    row["private_tests"]["input"],
                    row["private_tests"]["output"],
                ))[: self.cfg.max_cases]
                pub = list(zip(
                    row["public_tests"]["input"],
                    row["public_tests"]["output"],
                ))[: self.cfg.max_cases]
                tlim = row.get("time_limit", {}).get("seconds", 1)
                task = {
                    "name": row.get("name", ""),
                    "description": row.get("description", ""),
                    "public_tests": pub,
                    "private_tests": priv,
                    "time_limit": float(tlim),
                }
                if self.cfg.precompute_test_hashes:
                    h = hashlib.md5(task["name"].encode()).hexdigest()
                    self._hashes[h] = [hashlib.md5(f"{i}{o}".encode()).hexdigest() for i, o in priv]
                    task["_hash"] = h
                self.tasks.append(task)
            except Exception:
                continue
        random.shuffle(self.tasks)

    def sample(self) -> Dict[str, Any]:
        return random.choice(self.tasks)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Safe Code Execution with timeout, caching, parallelism
# ═══════════════════════════════════════════════════════════════════════════
class SafeCodeExecutor:
    _func_cache = SimpleLRUCache(max_size=2000)

    @staticmethod
    def extract_solve_fn(response: str) -> Optional[str]:
        key = hashlib.md5(response.encode()).hexdigest()
        cached = SafeCodeExecutor._func_cache.get(key)
        if cached:
            return cached
        m = re.search(r"```(?:python)?([\s\S]+?)```", response)
        src = m.group(1) if m else response
        fn_match = re.search(r"def solve\(.*?\):[\s\S]+", src)
        fn = fn_match.group(0) if fn_match else None
        if fn:
            SafeCodeExecutor._func_cache.put(key, fn)
        return fn

    @staticmethod
    def _run(code: str, inp: str, timeout: float, q) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(
                "import sys\n" + code +
                "\nif __name__=='__main__':\n    data=sys.stdin.read()\n    print(solve(data))"
            )
            path = f.name
        try:
            proc = subprocess.Popen(
                [sys.executable, path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
                preexec_fn=os.setsid
            )
            out, err = proc.communicate(inp, timeout=timeout)
            ok = proc.returncode == 0
            q.put((ok, out.strip() if ok else err.strip()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            q.put((False, "Timeout"))
        except Exception as e:
            q.put((False, str(e)))
        finally:
            try: os.remove(path)
            except: pass

    @staticmethod
    def execute(code: str, inp: str, timeout: float) -> Tuple[bool, str]:
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=SafeCodeExecutor._run,
            args=(code, inp, timeout, q)
        )
        p.start()
        p.join(timeout + 0.5)
        if not q.empty():
            ok, out = q.get()
        else:
            ok, out = False, "NoOutput"
        if p.is_alive():
            p.terminate(); p.join(0.1)
        return ok, out

# ═══════════════════════════════════════════════════════════════════════════
# 5. Module-level function for parallel execution (FIX)
# ═══════════════════════════════════════════════════════════════════════════

def _run_single_test(test_data: Tuple[int, str, str, str, float]) -> str:
    """
    Module-level function to run a single test case.
    This can be pickled for ProcessPoolExecutor.
    
    Args:
        test_data: (idx, code_str, inp, exp, timeout)
    
    Returns:
        str: Log line for this test case
    """
    idx, code_str, inp, exp, timeout = test_data
    ok, out = SafeCodeExecutor.execute(code_str, inp, timeout)
    out = out.strip()
    exp = str(exp).strip()
    
    if ok and out == exp:
        status = "PASS"
    else:
        status = "FAIL"
    
    return f"PUB TC{idx}: {status}, out={out!r}, exp={exp!r}"

# ═══════════════════════════════════════════════════════════════════════════
# 6. SHPPOEnv: Planner → Coder → Debugger with public/private tests
# ═══════════════════════════════════════════════════════════════════════════

type_obs = TypedDict("Obs", {"role": str, "visible_files": Dict[str, str], "hidden_state": Any})
type_act = TypedDict("Act", {"filename": str, "content": str})

class SHPPOEnv(ABC):
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.dataset = CodeContestDataset(cfg)
        self.roles = ["planner", "coder", "debugger"]
        self.step_idx = 0
        self.ep_dir: Optional[str] = None
        self.current_task: Dict[str, Any] = {}
        self.temp_dirs: List[str] = []  # Track temp directories for cleanup

    def reset(self) -> type_obs:
        self.step_idx = 0
        self.current_task = self.dataset.sample()
        
        # Create temporary directory for this episode
        if self.cfg.use_temp_dir:
            self.ep_dir = tempfile.mkdtemp(prefix=self.cfg.temp_dir_prefix)
            self.temp_dirs.append(self.ep_dir)
        else:
            # Fallback to old behavior if needed
            self.ep_dir = os.path.join(
                "/tmp/shppo_env",
                hashlib.md5(str(random.random()).encode()).hexdigest()
            )
            os.makedirs(self.ep_dir, exist_ok=True)
        
        # write problem
        prob = self.current_task["description"]
        with open(os.path.join(self.ep_dir, "problem.md"), 'w') as f:
            f.write(prob)
        return self._obs_for(self.roles[0])

    def cleanup(self):
        """Clean up temporary directories"""
        if self.cfg.cleanup_on_exit:
            for temp_dir in self.temp_dirs:
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        print(f"Warning: Failed to cleanup temp dir {temp_dir}: {e}")
            self.temp_dirs.clear()

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()

    def step(self, action: type_act) -> Tuple[type_obs, float, bool, dict]:
        if not self.ep_dir:
            raise RuntimeError("Env not reset")
        # write action file
        with open(os.path.join(self.ep_dir, action['filename']), 'w') as f:
            f.write(action['content'])
        self.step_idx += 1
        # after coder writes code.py, run public tests for debugger
        if self.step_idx == 2:
            self._run_public_tests()  # CHANGED FROM run_public_tests
        done = False; reward = 0.0
        # after debugger writes fixed_code.py, run private tests for metric
        if self.step_idx >= len(self.roles):
            reward = self._run_private_tests()
            done = True
        next_role = self.roles[self.step_idx] if not done else self.roles[-1]
        return self._obs_for(next_role), reward, done, {}

    def _obs_for(self, role: str) -> type_obs:
        visibility = {
            "planner": ["problem.md"],
            "coder":   ["problem.md","plan.md"],
            "debugger":["problem.md","plan.md","code.py","run_log.txt"],
        }
        files: Dict[str, str] = {}
        for fn in visibility.get(role, []):
            path = os.path.join(self.ep_dir or '', fn)
            if os.path.exists(path):
                with open(path) as f: files[fn] = f.read()
        return {"role": role, "visible_files": files, "hidden_state": []}

    def _run_public_tests(self):
        """Run public tests and write results to run_log.txt"""
        code_path = os.path.join(self.ep_dir or '', "code.py")
        if os.path.exists(code_path):
            self.run_public_tests(code_path, self.ep_dir or '')
        else:
            # Create empty run_log.txt if no code file exists
            with open(os.path.join(self.ep_dir or '', "run_log.txt"), 'w') as f:
                f.write("No code file found for public testing\n")
    
    def run_public_tests(self, code_file: str, work_dir: str) -> str:
        """
        Executes public test cases on the given code_file and writes run_log.txt
        into work_dir. Returns the full log as a string.
        """
        # Read the code (in case it's not already on disk)
        with open(code_file, 'r') as cf:
            code_str = cf.read()

        tests = self.current_task.get("public_tests", [])
        log_lines: List[str] = []
        passed = 0
        total = len(tests)

        # Parallel or serial execution
        if self.cfg.use_parallel_execution and len(tests) > 1:
            workers = self.cfg.max_workers or os.cpu_count() or 1
            
            # Prepare test data for parallel execution
            test_data_list = [
                (i, code_str, inp, outp, self.cfg.test_timeout)
                for i, (inp, outp) in enumerate(tests)
            ]
            
            with ProcessPoolExecutor(max_workers=workers) as exe:
                log_lines = list(exe.map(_run_single_test, test_data_list))
                
            # Count passed tests
            passed = sum(1 for line in log_lines if "PASS" in line)
        else:
            # Serial execution (fallback)
            for i, (inp, outp) in enumerate(tests):
                ok, out = SafeCodeExecutor.execute(code_str, inp, self.cfg.test_timeout)
                out = out.strip()
                exp = str(outp).strip()
                if ok and out == exp:
                    status = "PASS"
                    passed += 1
                else:
                    status = "FAIL"
                line = f"PUB TC{i}: {status}, out={out!r}, exp={exp!r}"
                log_lines.append(line)

        # Write the aggregated log into work_dir/run_log.txt
        os.makedirs(work_dir, exist_ok=True)
        log_path = os.path.join(work_dir, "run_log.txt")
        with open(log_path, "w") as lf:
            lf.write("\n".join(log_lines))

        # Optionally, you can return a summary or the full log
        summary = f"Passed {passed}/{total} public tests\n"
        return summary + "\n".join(log_lines)

    def _run_private_tests(self) -> float:
        code_path = os.path.join(self.ep_dir or '', "fixed_code.py")
        with open(code_path) as f: code_str = f.read()
        tests = self.current_task.get("private_tests", [])
        log_lines: List[str] = []
        passed = 0; total = len(tests)
        
        if self.cfg.use_parallel_execution and len(tests) > 1:
            workers = self.cfg.max_workers or os.cpu_count() or 1
            with ProcessPoolExecutor(max_workers=workers) as exe:
                futures = [exe.submit(SafeCodeExecutor.execute, code_str, inp, self.cfg.test_timeout)
                           for inp, _ in tests]
                for i, fut in enumerate(futures):
                    ok, out = fut.result()
                    exp = str(tests[i][1]).strip()
                    log_lines.append(f"PRI TC{i}: ok={ok}, out={out}, exp={exp}")
                    if ok and out == exp: passed += 1
        else:
            # Serial execution (fallback)
            for i, (inp, outp) in enumerate(tests):
                ok, out = SafeCodeExecutor.execute(code_str, inp, self.cfg.test_timeout)
                exp = str(outp).strip()
                log_lines.append(f"PRI TC{i}: ok={ok}, out={out}, exp={exp}")
                if ok and out == exp: passed += 1
                
        with open(os.path.join(self.ep_dir or '', "private_run_log.txt"), 'w') as lf:
            lf.write("\n".join(log_lines))
        return passed / total if total>0 else 0.0