# shppo_emergent.py
from __future__ import annotations

# ────────────────────────────────────── stdlib ─────────────────────────────────────
import itertools
import os
import random
import subprocess
import tempfile
import textwrap
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import textwrap

# ─────────────────────────────────── third-party ───────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import trange, tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import get_peft_model, LoraConfig
import re

warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")

# ───────────────────────────────── configuration ───────────────────────────────────
OBS_DIM = 256
GLOB_DIM = 256
N_AGENTS = 3 # Default number of agents for training
MAX_STEPS = 20
RNN_HIDDEN_SIZE = 64
LATENT_Z_DIM = 3
HET_LAYER_OUT_DIM = 128

ACTION_TEMPLATES: Tuple[str, ...] = (
    "plan-subgoal",        
    "rephrase-prompt",     
    "assess-subgoals",     

    "generate-code",       
    "optimize-code",       

    "self-review",         
    "patch-bug",           
    "unit-fix",            

     "noop",             
)

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TEMPLATES)}

PLANNER_ACTIONS = {
    ACTION_TO_IDX["plan-subgoal"],
    ACTION_TO_IDX["rephrase-prompt"],
    ACTION_TO_IDX["assess-subgoals"],
}

CODE_ACTIONS = {
    ACTION_TO_IDX["generate-code"],
    ACTION_TO_IDX["optimize-code"],
}

DEBUG_ACTIONS = {
    ACTION_TO_IDX["self-review"],
    ACTION_TO_IDX["patch-bug"],
    ACTION_TO_IDX["unit-fix"],
}


# ==============================================================================
# Utility functions -----------------------------------------------------------
# ==============================================================================
def cosine_dist(x: torch.Tensor) -> torch.Tensor:
    sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
    n = x.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=x.device)
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return (1 - sim[mask]).mean()

# ==============================================================================
# 1. Dataset loader ------------------------------------------------------------
# ==============================================================================
class CodeContestDataset:
    def __init__(self, split: str = "train") -> None:
        ds = load_dataset("deepmind/code_contests", split=split, trust_remote_code=True)
        self.tasks: List[Dict[str, Any]] = []
        for row in ds:
            ins, outs = row["public_tests"]["input"], row["public_tests"]["output"]
            self.tasks.append({
                "name": row["name"], 
                "prompt": row["description"], 
                "tests_public": list(zip(ins, outs)), 
                "observation_features": self._embed(row["description"]), 
            })

    @staticmethod
    def _embed(text: str, dim: int = OBS_DIM) -> np.ndarray:
        vec = np.zeros(dim, np.float32)
        for ch in text[: dim * 4]: 
            vec[ord(ch) % dim] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec 

    def sample(self) -> Dict[str, Any]:
        return random.choice(self.tasks)

# ==============================================================================
# 2. Unsafe Python executor ---------------------------------------------------
# ==============================================================================

def build_py_wrap(impl: str, fn_name: str) -> str:
    return textwrap.dedent(f"""
    {impl}

    import sys
    if __name__ == "__main__":
        input_data = sys.stdin.read()
        res = {fn_name}(input_data)
        if not isinstance(res, str):
            raise RuntimeError("Function must return a string")
        sys.stdout.write(res)
    """)


def find_function_name(code: str) -> Optional[str]:

    pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    match = re.search(pattern, code)
    return match.group(1) if match else None


def exec_solution(src: str, stdin: Any, timeout: float = 2.0) -> Tuple[str, str]:
    if not src.strip():
        return "", "No code provided"

    impl_block = src.strip()
    fn_name = find_function_name(impl_block)
    if not fn_name:
        return "", "No valid function found"
    wrapped = build_py_wrap(impl_block, fn_name)

    if isinstance(stdin, (bytes, bytearray)):
        input_str = stdin.decode("utf-8", errors="ignore")
    else:
        input_str = str(stdin)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(wrapped)
        tmp.flush()
        filename = tmp.name

    stdout, stderr = "", ""
    try:
        proc = subprocess.run(
            ["python", filename],
            input=input_str,
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        stdout, stderr = proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        stderr = "TimeoutExpired"
    except Exception as e:
        stderr = f"Subprocess error: {e}"
    finally:
        os.remove(filename)

    return stdout, stderr



def extract_python_code(raw_output: str) -> str:
    """
    Extracts Python code from LLM output and dedents it for execution.
    Handles code blocks, def/main detection, or returns fallback text.
    """
    raw_output = raw_output.strip()

    match = re.search(r"```python\s*\n(.*?)```", raw_output, re.DOTALL)
    if match:
        return textwrap.dedent(match.group(1)).strip()

    match = re.search(r"```\s*\n(.*?)```", raw_output, re.DOTALL)
    if match:
        return textwrap.dedent(match.group(1)).strip()

    lines = raw_output.splitlines()
    for i, line in enumerate(lines):
        if "def " in line or "main(" in line:
            return textwrap.dedent("\n".join(lines[i:])).strip()

    # 4. fallback
    return textwrap.dedent(raw_output)



# ==============================================================================
# 3. Three-agent environment -------------------------------------------------
# ==============================================================================
class CodeGenEnv:
    def __init__(
        self,
        split: str = "train",
        actor: MarCog | None = None,
        n_agents: int = N_AGENTS
    ) -> None:
        self.ds       = CodeContestDataset(split)
        self.n_agents = n_agents
        self.obs_dim, self.glob_dim = OBS_DIM, GLOB_DIM
        self.max_steps = MAX_STEPS
        self.actor    = actor

        self.current_prompt = ""  
        self.code           = ""
        self.current_plan   = ""
        self.pass_fraction  = 0.0
        self.last_error     = ""
        self.step_i         = 0

        self.reset()

    @staticmethod
    def _embed(text: str, dim: int) -> np.ndarray:
        return CodeContestDataset._embed(text, dim)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.task          = self.ds.sample()
        self.step_i        = 0
        self.code          = ""
        self.current_plan  = ""
        self.pass_fraction = 0.0
        self.last_error    = ""
        self.current_prompt = self.task["prompt"]

        o0   = self._make_obs()
        obs  = np.stack([o0] * self.n_agents)
        glob = self._embed(self.task["name"], self.glob_dim)

        if self.actor:
            self.actor.set_prompts([self.current_prompt] * self.n_agents)

        return obs, glob

    def _make_obs(self) -> np.ndarray:
        q = self.obs_dim // 3
        p = self._embed(self.current_prompt, q)
        c = self._embed(self.code, q)
        e = self._embed(self.last_error, self.obs_dim - 2*q)
        e[0] = self.pass_fraction
        return np.concatenate([p, c, e])

    def _eval(self) -> Tuple[float, float]:
        if not self.code.strip():
            self.last_error = "No code to execute"
            return 0.0, 0.0

        public_tests = self.task.get("tests_public", [])
        if not public_tests:
            self.last_error = ""
            return 1.0, 0.0

        total_time = 0.0
        ok = 0
        errs: List[str] = []

        for inp, out in public_tests:
            start = time.perf_counter()
            got, err = exec_solution(self.code, inp)
            total_time += time.perf_counter() - start
            if err:
                errs.append(err)
            elif got == out:
                ok += 1

        n = len(public_tests)
        pf = ok / n if n else 0.0
        self.last_error = "\n".join(dict.fromkeys(errs))[:1024] if errs else ""
        return pf, (total_time / n) if n else 0.0

    def step(self, acts: np.ndarray):
        self.step_i += 1
        outputs: List[Tuple[str, str]] = []
        dev = next(self.actor.model.parameters()).device

        for idx in acts:
            tmpl = ACTION_TEMPLATES[int(idx)]
            raw = self.actor.generate_action(
                self.current_prompt,
                self.code,
                self.current_plan,
                self.last_error,
                self.pass_fraction,
                dev,
                tmpl
            )

            if tmpl == "rephrase-prompt":
                if raw.strip() and not raw.startswith("# Skipped"):
                    self.current_prompt = raw.strip()
                output = raw

            elif tmpl == "plan-subgoal":
                if raw.strip() and not raw.startswith("# Skipped"):
                    self.current_plan = raw.strip()
                output = raw

            elif tmpl in ("generate-code", "optimize-code"):
                cleaned = extract_python_code(raw)
                if cleaned.strip() and cleaned.strip() != self.code:
                    self.code = cleaned.strip()
                output = cleaned or raw

            elif tmpl in ("self-review", "patch-bug", "unit-fix"):
                if raw.startswith("# Skipped") or raw.startswith("# Cannot"):
                    output = raw
                else:
                    cleaned = extract_python_code(raw)
                    if cleaned.strip() and cleaned.strip() != self.code:
                        self.code = cleaned.strip()
                    output = cleaned or raw

            else:  
                output = raw

            outputs.append((tmpl, output))

        if not self.code.strip():
            tqdm.write(f"[WARN] No code at step {self.step_i}, penalty.")
            reward_scalar = -0.2
            done = False
            obs  = np.stack([self._make_obs()] * self.n_agents)
            glob = self._embed(self.task["name"], self.glob_dim)
            return obs, glob, np.array([reward_scalar]*self.n_agents, np.float32), done, outputs
        
        old_pf   = self.pass_fraction
        old_code = self.code.strip()
        pf, _    = self._eval()
        self.pass_fraction = pf
        tqdm.write(str(self.task["prompt"]))
        tqdm.write(str(self.task["tests_public"]))
        tqdm.write(f"[DEBUG] step {self.step_i}, code:\n{self.code}")
        tqdm.write(f"[DEBUG] last_error:\n{self.last_error}")
        tqdm.write(f"[DEBUG] pass_fraction: {pf}")
        delta = pf - old_pf
        reward_scalar = 0.0

        if pf == 1.0:
            reward_scalar += 1.0
        elif delta > 0:
            reward_scalar += 0.3
        else:
            reward_scalar -= 0.01

        if not self.last_error.strip():
            reward_scalar += 0.05
        elif "SyntaxError" in self.last_error or "Timeout" in self.last_error:
            reward_scalar -= 0.05
        else:
            reward_scalar -= 0.01

        if self.code.strip() != old_code:
            reward_scalar += 0.02

        for idx in acts:
            if int(idx) in PLANNER_ACTIONS:
                reward_scalar += 0.01
            elif int(idx) in DEBUG_ACTIONS:
                reward_scalar += 0.015
        

        
        done  = (pf == 1.0) or (self.step_i >= self.max_steps)
        obs   = np.stack([self._make_obs()] * self.n_agents)
        glob  = self._embed(self.task["name"], self.glob_dim)
        reward = np.array([reward_scalar]*self.n_agents, np.float32)

        return obs, glob, reward, done, outputs

# ==============================================================================
# 4. Neural building blocks ---------------------------------------------------
# ==============================================================================
def ortho(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hid: int = 128, layers: int = 2):
        super().__init__()
        seq, d = [], din
        for _ in range(layers - 1):
            seq += [nn.Linear(d, hid), nn.ReLU()] 
            d = hid 
        seq.append(nn.Linear(d, dout)) 
        self.net = nn.Sequential(*seq) 
        self.net.apply(lambda m: ortho(m, math.sqrt(2))) 

    def forward(self, x):
        return self.net(x)

class LatentNet(nn.Module):
    def __init__(self, od: int, hd: int, zdim: int = LATENT_Z_DIM):
        super().__init__()
        self.enc = MLP(od + hd, zdim * 2, hid=64, layers=3) 
        self.zdim = zdim 

    def forward(self, obs, mem):
        enc_out = self.enc(torch.cat([obs, mem], -1))
        mu, logv = enc_out.chunk(2, -1) 
        sig = logv.clamp(-6, 2).exp() 
        z = mu + sig * torch.randn_like(mu) 
        return z, mu, sig 

class InfNet(nn.Module):
    def __init__(self, zdim: int = LATENT_Z_DIM, gdim: int = GLOB_DIM):
        super().__init__()
        self.v = MLP(gdim + 2 * zdim, 1, hid=128, layers=3) 

    def forward(self, g, mu, sig):
        return self.v(torch.cat([g, mu, sig], -1)).squeeze(-1) 

class Decoder(nn.Module):
    def __init__(self, zdim: int, hdim: int, od: int):
        super().__init__()
        self.fc_w = MLP(zdim, hdim * od, hid=zdim * 2, layers=2) 
        # MLP to generate bias vector (b)
        self.fc_b = MLP(zdim, od, hid=zdim, layers=2) 
        self.hdim, self.od = hdim, od # Store input and output dimensions
        self.fc_w.apply(lambda m: ortho(m, math.sqrt(0.1)))
        self.fc_b.apply(lambda m: ortho(m, math.sqrt(0.1)))

    def forward(self, z):
        B = z.size(0) # Batch size
        W = self.fc_w(z).view(B, self.od, self.hdim) 
        b = self.fc_b(z) 
        return W, b 

class MultiHeadCritic(nn.Module):
    def __init__(self, glob_dim: int, n_agents: int):
        super().__init__()
        self.n_agents = n_agents
        self.proj = MLP(glob_dim, 64, hid=32, layers=2)
        self.heads = nn.ModuleList([
            MLP(64, 1, hid=32, layers=2) for _ in range(n_agents)
        ])

    def forward(self, glob: torch.Tensor, agent_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.proj(glob)  # (B, H)

        if agent_ids is not None:
            B = glob.size(0)
            assert agent_ids.dim() == 1 and agent_ids.size(0) == B, \
                f"agent_ids must be (B,) tensor with B={B}, got {agent_ids.shape}"

            outputs = []
            for i in range(self.n_agents):
                mask = (agent_ids == i)
                if mask.any():
                    out = self.heads[i](h[mask]).squeeze(-1)  # (M,)
                    outputs.append((mask, out))

            values = torch.zeros(B, device=glob.device)
            for mask, out in outputs:
                values[mask] = out
            return values  # shape: (B,)
        else:
            return torch.cat([head(h).unsqueeze(1) for head in self.heads], dim=1)


class MarCog(nn.Module):
    def __init__(self, repo: str = "./models/Qwen2.5-Coder-32B-Instruct", n_agents: int = N_AGENTS):
        super().__init__()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(repo, device_map="auto" if torch.cuda.is_available() else None,
                                                    quantization_config=bnb, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        peft_cfg = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.05)
        self.model = get_peft_model(base, peft_cfg)

        self.n_agents = n_agents
        self._last_prompts: List[str] = []
        self._pooled_features: Optional[torch.Tensor] = None

        self.feat_dim = OBS_DIM + RNN_HIDDEN_SIZE + base.config.hidden_size
        self.decoder = Decoder(LATENT_Z_DIM, self.feat_dim, len(ACTION_TEMPLATES))
        llh = base.config.hidden_size
        self.head_mlp = MLP(llh + HET_LAYER_OUT_DIM, len(ACTION_TEMPLATES), hid=256, layers=2)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def set_prompts(self, prompts: List[str]):
        prompts = prompts[: self.n_agents]
        if prompts == self._last_prompts:
            return
        self._last_prompts = prompts
        toks = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**toks, output_hidden_states=True, return_dict=True)
        self._pooled_features = out.hidden_states[-1].mean(dim=1)

    def forward(self, obs: torch.Tensor, mem: torch.Tensor, z: torch.Tensor, agent_ids: Optional[torch.Tensor] = None) -> Categorical:
        if self._pooled_features is None:
            raise RuntimeError("Call set_prompts() before forward()")

        B = mem.size(0)
        N = self._pooled_features.size(0)

        if agent_ids is None:
            agent_ids = torch.arange(B, device=self.device) % N  

        pooled = self._pooled_features[agent_ids] 
        feat = torch.cat([pooled, obs, mem], dim=-1)
        assert feat.shape[1] == self.decoder.hdim, f"Expected feat_dim={self.decoder.hdim}, got {feat.shape[1]}"

        W, b = self.decoder(z)  
        logits = torch.bmm(W, feat.unsqueeze(-1)).squeeze(-1) + b
        return Categorical(logits=logits)

    @torch.no_grad()
    def generate_action(
        self,
        prompt: str,
        prev_code: str,
        prev_plan: str,
        errs: str,
        pf: float,
        device: torch.device,
        action_template: str
    ) -> str:
        # 2) 프롬프트 본문(body) 결정
        if action_template == "plan-subgoal":
            body = (
                f"Task:\n{prompt}\n\n"
                f"Previous plan:\n{prev_plan}\n\n"
                f"Pass fraction: {pf*100:.1f}%\n"
                f"Errors:\n{errs}\n\n"
                f"Action: {action_template}\n"
                f"You have to {action_template}. Produce a sequence of bullet-point plan and subgoals.\n"
            )

        elif action_template == "rephrase-prompt":
            body = (
                f"Original Prompt:\n{prompt}\n\n"
                f"Action: {action_template}\n"
                f"You have to {action_template}. Rewrite the task description to make it clearer and more precise.\n"
            )

        elif action_template == "assess-subgoals":
            if not prev_plan.strip():
                return "# Skipped assess-subgoals: no current plan available."
            body = (
                f"Task:\n{prompt}\n\n"
                f"Subgoal plan:\n{prev_plan}\n"
                f"Pass fraction: {pf*100:.1f}%\n"
                f"Errors:\n{errs}\n\n"
                f"Action: {action_template}\n"
                f"You have to {action_template}. Evaluate whether the subgoals are logically valid, sufficient, and well-structured.\n"
                "Suggest improvements if needed.\n"
            )

        elif action_template == "generate-code":
            body = (
                f"Task:\n{prompt}\n\n"
                f"Subgoal plan:\n{prev_plan}\n"
                f"Pass fraction: {pf*100:.1f}%\n"
                f"Errors:\n{errs}\n\n"
                f"Action: {action_template}\n"
                f"You have to {action_template}. Implement a function with this format:\n"
                "    def solve(input_data: str) -> str:\n"
                "        \"\"\"\n"
                "        input_data: the entire stdin as one string\n"
                "        Returns: exactly what should be printed to stdout\n"
                "        \"\"\"\n"
                "        # parse input_data\n"
                "        # compute answer\n"
                "        # return result string\n\n"
                "Requirements:\n"
                "- Do NOT use input() or print() inside your function.\n"
                "- Do NOT include any comments or extra text—only the function definition.\n"
            )

        elif action_template == "optimize-code":
            if not prev_code.strip():
                return "# Skipped optimize-code: no code to optimize."
            body = (
                f"Task:\n{prompt}\n\n"
                f"Subgoal plan:\n{prev_plan}\n"
                f"Pass fraction: {pf*100:.1f}%\n"
                f"Errors:\n{errs}\n\n"
                f"Action: {action_template}\n"
                f"You have to {action_template}. Optimize the code to improve time and space complexity without changing its functionality.\n"
            )

        elif action_template == "self-review":
            if not prev_code.strip():
                return "# Skipped self-review: no code to review."
            body = (
                f"Task:\n{prompt}\n\n"
                f"Code for review:\n{prev_code}\n"
                f"Pass fraction: {pf*100:.1f}%\n"
                f"Errors:\n{errs}\n\n"
                f"Action: {action_template}\n"
                f"You have to {action_template}. Identify any possible logical errors, bad practices, or things to improve in the above code.\n"
            )

        elif action_template == "patch-bug":
            if not prev_code.strip():
                return "# Skipped patch-bug: no code to debug."
            body = (
                f"Task:\n{prompt}\n\n"
                f"Code needing fix:\n{prev_code}\n\n"
                f"Errors:\n{errs}\n\n"
                f"Action: {action_template}\n"
                f"You have to {action_template}. Please correct the given code by solving the errors above and output the full updated version only.\n"
            )

        elif action_template == "unit-fix":
            if not prev_code.strip():
                return "# Skipped unit-fix: no code to fix."
            body = (
                f"Task:\n{prompt}\n\n"
                f"Failing Code:\n{prev_code}\n\n"
                f"Pass fraction: {pf*100:.1f}%\n"
                f"Test Failures / Errors:\n{errs}\n\n"
                f"Action: {action_template}\n"
                f"You have to {action_template}. Fix the code so that it passes all the tests. Only output the corrected code.\n"
            )

        elif action_template == "noop":
            return "# No operation selected for this step."

        else:
            raise ValueError(f"Unknown action_template: {action_template}")
        toks = self.tokenizer(
            body,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).to(device)
        gen_cfg = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,
            no_repeat_ngram_size=2,
            repetition_penalty=1.1,
        )
        out = self.model.generate(
            **toks,
            generation_config=gen_cfg,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            out[0][toks.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()
        


@dataclass
class CFG:
    gamma: float = 0.95 # Discount factor
    lam:   float = 0.95 # GAE lambda parameter
    clip:  float = 0.2 # PPO clipping parameter
    lr:    float = 5e-4 # Learning rate for Actor, RNN, LatentNet, Decoder, Head MLP
    lr_inf:float = 5e-3 # Learning rate for InferenceNet
    ent:   float = 5e-2 # Entropy bonus coefficient
    lambda_V_I: float = 1.0 # Weight for InferenceNet value prediction loss in Actor objective (maximize V_I)
    le:     float = 1e-2   # Weight for Latent Entropy loss in Actor objective (minimize Le)
    ld:     float = 1e-2   # Weight for Latent Distance loss in Actor objective (minimize -Ld = maximize Ld)
    lv_inf: float = 1e-1   # Weight for InferenceNet's own loss (MSE against return)
    me:     int   = 4 # Number of PPO epochs per update
    bs:     int   = 64 # Batch size for PPO update mini-batches
    max_g: float = 0.5 # Max gradient norm for clipping

class Buffer:
    def __init__(self, T:int, n:int, od:int, gd:int, hd:int):
        self.T, self.n = T, n # T: max episode steps, n: number of agents
        self.obs = torch.zeros(T, n, od) # Observations (T, N_AGENTS, OBS_DIM)
        self.mem = torch.zeros(T, n, hd) # RNN hidden states (T, N_AGENTS, RNN_HIDDEN_SIZE)
        self.lat = torch.zeros(T, n, LATENT_Z_DIM) # Sampled latent variables (T, N_AGENTS, LATENT_Z_DIM)
        self.mu  = torch.zeros(T, n, LATENT_Z_DIM) # Latent means (T, N_AGENTS, LATENT_Z_DIM)
        self.sig = torch.zeros(T, n, LATENT_Z_DIM) # Latent std dev (T, N_AGENTS, LATENT_Z_DIM)
        self.act = torch.zeros(T, n, dtype=torch.long) # Actions (T, N_AGENTS)
        self.logp= torch.zeros(T, n) # Log probabilities of chosen actions (T, N_AGENTS)
        self.rew = torch.zeros(T, n) # Rewards (T, N_AGENTS)
        self.val = torch.zeros(T, n) # Value predictions (T, N_AGENTS)
        self.glob= torch.zeros(T, gd) # Global states (T, GLOB_DIM) - stored once per timestep
        self.done= torch.zeros(T, dtype=torch.bool) # Done flags (T,) - stored once per timestep
        self.ptr = 0 # Pointer to the current position in the buffer

    def add(self, o, m, z, mu, sig, a, lp, r, v, g, d):
        if self.ptr >= self.T:
            warnings.warn("Buffer is full. Increase buffer size or implement cyclic buffer.")
            return
        i = self.ptr
        self.obs[i]=o; self.mem[i]=m; self.lat[i]=z; self.mu[i]=mu; self.sig[i]=sig
        self.act[i]=a; self.logp[i]=lp; self.rew[i]=r; self.val[i]=v.squeeze()
        self.glob[i]=g; self.done[i]=d
        self.ptr+=1
        
    def data(self):
        return {k: getattr(self, k)[:self.ptr] for k in [
            'obs','mem','lat','mu','sig','act','logp','rew','val','glob','done'
        ]}

    def clear(self):
        self.ptr=0
    
class Trainer:
    def __init__(self, env: CodeGenEnv, cfg: CFG):
        self.env, self.cfg = env, cfg 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rnn     = nn.GRU(OBS_DIM, RNN_HIDDEN_SIZE, batch_first=True).to(self.device) 
        self.actor   = MarCog().to(self.device) 
        self.lat_net = LatentNet(OBS_DIM, RNN_HIDDEN_SIZE).to(self.device) 
        self.crit = MultiHeadCritic(GLOB_DIM, env.n_agents).to(self.device) 
        self.inf     = InfNet().to(self.device) 

        self.optA = torch.optim.Adam(
            itertools.chain(
                self.actor.model.parameters(), 
                self.actor.decoder.parameters(),
                self.actor.head_mlp.parameters(), 
                self.rnn.parameters(), 
                self.lat_net.parameters(), 
            ), lr=cfg.lr 
        )
        self.optC = torch.optim.Adam(self.crit.parameters(), lr=cfg.lr) 
        self.optI = torch.optim.Adam(self.inf.parameters(), lr=cfg.lr_inf) 
        self.optL = torch.optim.Adam(self.lat_net.parameters(), lr=cfg.lr)
        self.buf = Buffer(self.env.max_steps, self.env.n_agents, OBS_DIM, GLOB_DIM, RNN_HIDDEN_SIZE) 
        self.env.actor = self.actor 

    def collect(self) -> float:
        self.buf.clear()
        obs, glob = self.env.reset()
        current_mem_rnn = torch.zeros(1, self.env.n_agents, RNN_HIDDEN_SIZE, device=self.device)
        self.actor.set_prompts([self.env.task["prompt"]] * self.env.n_agents)

        for t in range(self.env.max_steps):
            o_t = torch.tensor(obs, device=self.device, dtype=torch.float32)
            g_t = torch.tensor(glob, device=self.device, dtype=torch.float32)
            g_t_expanded = g_t.unsqueeze(0).repeat(self.env.n_agents, 1)

            rnn_out, next_mem_rnn = self.rnn(o_t.unsqueeze(1), current_mem_rnn)
            mem_for_actor = rnn_out.squeeze(1)

            with torch.no_grad():
                z, mu, sig = self.lat_net(o_t, mem_for_actor)
                dist = self.actor(o_t, mem_for_actor, z)
                a = dist.sample()
                lp = dist.log_prob(a)
                agent_ids = torch.arange(self.env.n_agents, device=self.device)
                v = self.crit(g_t_expanded, agent_ids=agent_ids)  # shape: (N,)

            obs_next, glob_next, rew, done, outputs = self.env.step(a.cpu().numpy())

            print(f"Step {t+1}:")
            for agent_id, (tmpl, out) in enumerate(outputs, start=1):
                snippet = out.replace("\n", " ")[:100]
                tqdm.write(f"  Agent{agent_id} [{tmpl}]: {snippet}...")
            print()

            self.buf.add(
                o_t.cpu(), mem_for_actor.cpu(), z.cpu(), mu.cpu(), sig.cpu(),
                a.cpu(), lp.cpu(), torch.tensor(rew, dtype=torch.float32), v.cpu(),
                g_t.cpu(), bool(done)
            )

            obs, glob = obs_next, glob_next
            current_mem_rnn = next_mem_rnn

            if done:
                break

        final_glob_tensor = torch.tensor(glob, device=self.device, dtype=torch.float32).unsqueeze(0)
        last_val = self.crit(final_glob_tensor).mean().item()
        return last_val


    def adv_ret(self, data, last_v):
        rews   = data['rew'].to(self.device)   
        vals   = data['val'].to(self.device)   
        dones  = data['done'].to(self.device)  

        T, N = rews.shape 
        adv = torch.zeros(T, N, device=self.device) 
        gae = torch.zeros(N, device=self.device) 

        next_val = torch.full((N,), last_v, device=self.device) 

        for t in reversed(range(T)):
            mask_t = 1.0 - dones[t].float() 
            delta = rews[t] + self.cfg.gamma * next_val * mask_t - vals[t] 

            gae = delta + self.cfg.gamma * self.cfg.lam * mask_t * gae     
            adv[t] = gae 
            
            next_val = vals[t] 

        returns = adv + vals

        return adv.cpu(), returns.cpu()
    
    def ppo_update(self, data, adv, ret):
        cfg, dev = self.cfg, self.device
        T, N = data["obs"].shape[:2]
        M = T * N

        obs_flat    = data["obs"].reshape(M, -1).to(dev)
        mem_flat    = data["mem"].reshape(M, -1).to(dev)
        act_flat    = data["act"].reshape(M).to(dev)
        oldlp_flat  = data["logp"].reshape(M).to(dev)
        adv_flat    = adv.reshape(M).to(dev)
        ret_flat    = ret.reshape(M).to(dev)
        glob_flat   = data["glob"].unsqueeze(1).repeat(1, N, 1).reshape(M, -1).to(dev)
        agent_ids   = torch.arange(N, device=dev).expand(T, -1).reshape(M)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        for _ in range(cfg.me):                      
            perm = torch.randperm(M, device=dev)
            for start in range(0, M, cfg.bs):
                mb       = perm[start : start + cfg.bs]
                o_mb     = obs_flat[mb].detach()      
                m_mb     = mem_flat[mb].detach()
                a_mb     = act_flat[mb]
                oldlp_mb = oldlp_flat[mb]
                adv_mb   = adv_flat[mb]
                ret_mb   = ret_flat[mb]
                g_mb     = glob_flat[mb]
                id_mb    = agent_ids[mb]

                # --------------------------
                # 1) Policy (Actor) update
                # --------------------------
                with torch.no_grad():
                    z_pol, mu_pol, sig_pol = self.lat_net(o_mb, m_mb)

                dist     = self.actor(o_mb, m_mb, z_pol, agent_ids=id_mb)
                lp       = dist.log_prob(a_mb)
                entropy  = dist.entropy().mean()
                ratio    = torch.exp(lp - oldlp_mb)
                s1       = ratio * adv_mb
                s2       = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_mb
                loss_surr= -torch.min(s1, s2).mean()

                Le_pol = (0.5 * math.log(2 * math.pi * math.e) + torch.log(sig_pol + 1e-8)).mean()
                Ld_pol = cosine_dist(z_pol)

                loss_actor = loss_surr \
                            - cfg.ent * entropy \
                            + cfg.le * Le_pol \
                            - cfg.ld * Ld_pol

                self.optA.zero_grad()
                loss_actor.backward()
                nn.utils.clip_grad_norm_(itertools.chain(
                    self.actor.model.parameters(),
                    self.actor.decoder.parameters(),
                    self.actor.head_mlp.parameters(),
                    self.rnn.parameters(),
                    self.lat_net.parameters(),
                ), cfg.max_g)
                self.optA.step()

                # --------------------------
                # 2) Critic update
                # --------------------------
                v_pred   = self.crit(g_mb, agent_ids=id_mb).squeeze(-1)
                loss_crit= F.mse_loss(v_pred, ret_mb)

                self.optC.zero_grad()
                loss_crit.backward()
                nn.utils.clip_grad_norm_(self.crit.parameters(), cfg.max_g)
                self.optC.step()

                # --------------------------
                # 3) Latent network update
                # --------------------------
                z_lat, mu_lat, sig_lat = self.lat_net(o_mb, m_mb)
                Le_lat = (0.5 * math.log(2 * math.pi * math.e) + torch.log(sig_lat + 1e-8)).mean()
                Ld_lat = cosine_dist(z_lat)
                VIl    = self.inf(g_mb, mu_lat, sig_lat)

                loss_latent = cfg.lambda_V_I * F.mse_loss(VIl, ret_mb) \
                            + cfg.le * Le_lat \
                            - cfg.ld * Ld_lat

                self.optL.zero_grad()
                loss_latent.backward()
                nn.utils.clip_grad_norm_(self.lat_net.parameters(), cfg.max_g)
                self.optL.step()

                # --------------------------
                # 4) InferenceNet update
                # --------------------------
                V_inf   = self.inf(g_mb, mu_lat.detach(), sig_lat.detach())
                loss_inf= F.mse_loss(V_inf, ret_mb) * cfg.lv_inf

                self.optI.zero_grad()
                loss_inf.backward()
                nn.utils.clip_grad_norm_(self.inf.parameters(), cfg.max_g)
                self.optI.step()

    

    def train(self, episodes: int):
        for ep in trange(episodes, desc="Training Episodes"):
            last_v = self.collect() 
            data = self.buf.data()            
            if data['obs'].shape[0] == 0:
                print(f"Episode {ep}: No data collected, skipping PPO update.")
                continue

            adv, ret = self.adv_ret(data, last_v)

            self.ppo_update(data, adv, ret)
            
            total_r_episode = data["rew"].sum().item()
            avg_pass_frac = self.env.pass_fraction 

            tqdm.write(f"Episode {ep}: Total Reward: {total_r_episode:.2f}, Final Pass Fraction: {avg_pass_frac:.2f}")

    def evaluate(self, test_agent_counts: List[int] = [3, 5, 7], num_eval_episodes: int = 10):
        """
        Evaluates the trained actor on environments with different numbers of agents
        to test zero-shot scalability.
        """
        print("\nStarting Zero-Shot Scalability Evaluation...")
        self.actor.eval()
        self.rnn.eval()
        self.lat_net.eval()
        self.crit.eval()
        self.inf.eval()
        with torch.no_grad(): 
            for n in test_agent_counts:
                print(f"\nEvaluating with {n} agents...")
                eval_env = CodeGenEnv(split="train", actor=self.actor, n_agents=n)

                total_final_pass_fraction = 0.0
                total_reward_across_episodes = 0.0

                for eval_ep in tqdm(range(num_eval_episodes), desc=f"Eval {n} agents"):
                    obs, glob = eval_env.reset()
                    self.actor.set_prompts([eval_env.task["prompt"]] * n) 
                    
                    current_mem_rnn = torch.zeros(1, n, RNN_HIDDEN_SIZE, device=self.device) 
                    episode_reward = 0.0

                    for step_i in range(eval_env.max_steps):
                        o_t = torch.tensor(obs, device=self.device, dtype=torch.float32) 
                        g_t = torch.tensor(glob, device=self.device, dtype=torch.float32) 
                        
                        rnn_out, next_mem_rnn = self.rnn(o_t.unsqueeze(1), current_mem_rnn)
                        mem_for_actor = rnn_out.squeeze(1) 

                        z, mu, sig = self.lat_net(o_t, mem_for_actor) 
                        dist = self.actor(o_t, mem_for_actor, z) 

                        a = dist.sample()  
                        obs_next, glob_next, rew, done, _ = eval_env.step(a.cpu().numpy()) 

                        episode_reward += rew.mean().item() 
                        obs = obs_next
                        glob = glob_next
                        current_mem_rnn = next_mem_rnn 

                        if done:
                            break 

                    total_final_pass_fraction += eval_env.pass_fraction
                    total_reward_across_episodes += episode_reward

                avg_final_pass_fraction = total_final_pass_fraction / num_eval_episodes
                avg_episode_reward = total_reward_across_episodes / num_eval_episodes

                print(f"Average Final Pass Fraction: {avg_final_pass_fraction:.2f}")
                print(f"Average Total Episode Reward: {avg_episode_reward:.2f}")

        self.actor.train()
        self.rnn.train()
        self.lat_net.train()
        self.crit.train()
        self.inf.train()


if __name__ == "__main__":
    episodes = 100
    split    = "train"
    cfg      = CFG()
    env = CodeGenEnv(split=split, actor=None, n_agents=N_AGENTS)
    trainer  = Trainer(env, cfg)
    
    print(f"Starting training for {episodes} episodes with {N_AGENTS} agents...")

    try:
        trainer.train(episodes)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback; traceback.print_exc()
    finally:
        save_dir = "shppo_emergent"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving models to '{save_dir}'...")
        try:
            trainer.actor.model.save_pretrained(save_dir)
            trainer.actor.tokenizer.save_pretrained(save_dir)
        
            torch.save(trainer.lat_net.state_dict(),  os.path.join(save_dir, "latent_net.pth"))
            torch.save(trainer.rnn.state_dict(),      os.path.join(save_dir, "rnn.pth"))
            torch.save(trainer.crit.state_dict(),     os.path.join(save_dir, "critic.pth"))
            torch.save(trainer.inf.state_dict(),      os.path.join(save_dir, "inf_net.pth"))
            torch.save(trainer.actor.decoder.state_dict(),   os.path.join(save_dir, "decoder.pth"))
            torch.save(trainer.actor.head_mlp.state_dict(), os.path.join(save_dir, "head_mlp.pth"))

            print("All models saved.")
        except Exception as e:
            print(f"Error during model saving: {e}")

       