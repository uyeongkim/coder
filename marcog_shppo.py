# marcog_shppo.py — SHPPO on DeepMind *code_contest*
# Fine-tunes Qwen2.5-Coder-32B with LoRA and logs to Weights & Biases
from __future__ import annotations

# ────────────────────────────────────── stdlib ─────────────────────────────────────
import argparse
import itertools
import json
import math
import os
import random
import subprocess
import tempfile
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ─────────────────────────────────── third-party ───────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import trange
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")

# ───────────────────────────────── configuration ───────────────────────────────────
OBS_DIM:  int = 256
GLOB_DIM: int = 256
N_AGENTS: int = 3
MAX_STEPS: int = 8      # environment steps per episode

ACTION_TEMPLATES: Tuple[str, ...] = (
    "<planner>", "<coder>", "<debugger>",
    "plan-subgoal", "rephrase-prompt", "write-tests",
    "gen-code-v1", "gen-code-v2", "self-review", "patch-bug",
    "unit-fix", "noop",
)

# ==============================================================================
# 1. Dataset loader (🤗) --------------------------------------------------------
# ==============================================================================
class CodeContestDataset:
    """Thin wrapper around the DeepMind *code_contest* dataset."""

    def __init__(self, split: str = "train") -> None:
        ds = load_dataset("deepmind/code_contest", split=split, trust_remote_code=True)
        self.tasks: List[Dict[str, Any]] = []

        for row in ds:
            inputs, outputs = row["public_tests"]["input"], row["public_tests"]["output"]
            self.tasks.append(
                {
                    "name": row["name"],
                    "prompt": row["description"],
                    "tests_public": list(zip(inputs, outputs)),
                }
            )

    def sample(self) -> Dict[str, Any]:
        """Return a random task."""
        return random.choice(self.tasks)

# ==============================================================================
# 2. Unsafe Python executor (placeholder) --------------------------------------
# ==============================================================================
PY_WRAP = (
    "{impl}\n"
    "if __name__=='__main__':\n"
    "    import sys\n"
    "    data = sys.stdin.read()\n"
    "    print(main(data))"
)

def exec_solution(src: str, stdin: str, timeout: float = 2.0) -> str:
    """Run user code in a subprocess and capture stdout (very unsafe – demo only)."""
    wrapped = PY_WRAP.format(impl=textwrap.indent(src, ""))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(wrapped)
        tmp.flush()
        filename = tmp.name

    try:
        proc = subprocess.run(
            ["python", filename],
            input=stdin.encode(),
            capture_output=True,
            timeout=timeout,
        )
        return proc.stdout.decode().strip()
    except subprocess.TimeoutExpired:
        return ""
    finally:
        os.remove(filename)

# ==============================================================================
# 3. Three-agent environment ----------------------------------------------------
# ==============================================================================
class CodeGenEnv:
    """Three-agent loop: Planner (0), Coder (1), Debugger (2)."""

    def __init__(self, split: str = "train") -> None:
        self.ds = CodeContestDataset(split)
        self.n_agents = N_AGENTS
        self.obs_dim  = OBS_DIM
        self.glob_dim = GLOB_DIM
        self.max_steps = MAX_STEPS
        self.reset()

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _bag_embed(text: str, dim: int) -> np.ndarray:
        vec = np.zeros(dim, np.float32)
        for ch in text[: dim * 4]:
            vec[ord(ch) % dim] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    # ── gym-like API ─────────────────────────────────────────────────────────
    def reset(self):
        self.task = self.ds.sample()
        self.step_i = 0
        self.code = ""
        obs = np.stack(
            [self._bag_embed(self.task["prompt"], self.obs_dim)] * self.n_agents
        )
        glob = self._bag_embed(self.task["name"], self.glob_dim)
        return obs, glob

    # ------------------------------------------------------------------
    def _eval_tests(self) -> Tuple[float, float]:
        """Return (pass_fraction, avg_time_seconds) for public tests."""
        import time

        total_t, ok = 0.0, 0
        if not self.code:
            return 0.0, 0.0

        for inp, out in self.task["tests_public"]:
            st = time.perf_counter()
            res = exec_solution(self.code, inp)
            total_t += time.perf_counter() - st
            if res == out:
                ok += 1

        n = len(self.task["tests_public"])
        pf   = ok / n if n else 0.0
        avg_t = total_t / n if n else 0.0
        return pf, avg_t

    def _pass_frac(self) -> float:
        pf, _ = self._eval_tests()
        return pf

    # ------------------------------------------------------------------
    def step(self, acts: np.ndarray):
        self.step_i += 1

        # Trivial coding template
        if acts[1] == 1 and not self.code:
            self.code = "def main(data):\n    return data"

        pf, avg_t = self._eval_tests()
        time_score = max(0.0, 1.0 - avg_t / 2.0)  # 0–1 scaling

        reward = np.array(
            [
                0.05 * pf + 0.05 * time_score,  # Planner
                0.70 * pf + 0.30 * time_score,  # Coder
                0.10 * pf + 0.10 * time_score,  # Debugger
            ],
            np.float32,
        )

        done = pf == 1.0 or self.step_i >= self.max_steps
        obs = np.stack(
            [self._bag_embed(self.task["prompt"], self.obs_dim)] * self.n_agents
        )
        glob = self._bag_embed(self.task["name"], self.glob_dim)
        return obs, glob, reward, done, {}

# ==============================================================================
# 4. Neural building blocks ----------------------------------------------------
# ==============================================================================
def ortho(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
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

    def forward(self, x):  # noqa: D401
        return self.net(x)

class LatentNet(nn.Module):
    """Encodes (obs, mem) → latent 𝑧 ~ 𝓝(μ, σ²)."""

    def __init__(self, obs_dim: int, mem_dim: int, zdim: int = 3) -> None:
        super().__init__()
        self.enc = MLP(obs_dim + mem_dim, zdim * 2, hid=64, layers=3)
        self.zdim = zdim

    def forward(
        self, obs: torch.Tensor, mem: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.enc(torch.cat([obs, mem], dim=-1)).chunk(2, dim=-1)
        sigma = log_var.clamp(-6, 2).exp()
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        return z, mu, sigma

class InfNet(nn.Module):
    def __init__(self, n_agents: int, zdim: int, gdim: int):
        super().__init__()
        self.v = MLP(gdim + n_agents * zdim * 2, 1, hid=128, layers=3)

    def forward(
        self, g: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor
    ) -> torch.Tensor:
        flat = torch.cat([g, mu.flatten(1), sig.flatten(1)], dim=-1)
        return self.v(flat).squeeze(-1)

# ---- LLM-based Actor ------------------------------------------------------
class LLMActor(nn.Module):
    """LoRA-tuned Qwen2.5-Coder → action logits conditioned on latent 𝑧."""

    def __init__(self, repo: str = "Qwen/Qwen2.5-Coder-32B", zdim: int = 3) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            repo,
            device_map="auto",
            load_in_8bit=True,  # change to 4-bit if GPU constrained
            trust_remote_code=True,
        )

        # tiny LoRA adapter
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05
        )
        self.model = get_peft_model(base, lora_cfg)

        hidden = base.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden + zdim, 256),
            nn.ReLU(),
            nn.Linear(256, len(ACTION_TEMPLATES)),
        )
        self.head.apply(lambda m: ortho(m, math.sqrt(2)))

        # freeze base weights
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.peft_parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

        self._prompts: List[str] = []

    # ------------------------------------------------------------------
    def set_prompts(self, prompts: List[str]):
        self._prompts = prompts

    @torch.no_grad()
    def _pool(self, texts: List[str], device: torch.device) -> torch.Tensor:
        toks = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outs = self.model.base_model.model(
            **toks, output_hidden_states=True, return_dict=True
        )
        return outs.hidden_states[-1].mean(1)  # (B, hidden)

    def forward(
        self,
        obs: torch.Tensor,  # unused but keeps signature aligned with other actors
        mem: torch.Tensor,
        z: torch.Tensor,
    ):
        if not self._prompts:
            raise RuntimeError("LLMActor.set_prompts() must be called before forward")
        pooled = self._pool(self._prompts, obs.device)
        logits = self.head(torch.cat([pooled, z], dim=-1))
        dist = Categorical(logits=logits)
        h_next = torch.zeros_like(mem)  # latent memory not used here
        return dist, h_next

# ---- Helper for diversity loss -------------------------------------------
def cosine_dist(z: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
    return (1 - cos).mean()

# ==============================================================================
# 5. Buffer & SHPPO trainer ----------------------------------------------------
# ==============================================================================
@dataclass
class CFG:
    gamma: float = 0.95
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 5e-4
    lr_inf: float = 5e-3
    ent: float = 1e-2
    vf: float = 0.5
    max_g: float = 0.5
    le: float = 1e-3
    ld: float = 1e-2
    lv: float = 1e-1
    me: int = 4
    bs: int = 256

class Buffer:
    def __init__(self, T: int, n: int, od: int, gd: int):
        self.T, self.n = T, n
        self.obs  = torch.zeros(T, n, od)
        self.lat  = torch.zeros(T, n, 3)
        self.mu   = torch.zeros(T, n, 3)
        self.sig  = torch.zeros(T, n, 3)
        self.act  = torch.zeros(T, n, dtype=torch.long)
        self.logp = torch.zeros(T, n)
        self.rew  = torch.zeros(T, n)
        self.val  = torch.zeros(T, n)
        self.glob = torch.zeros(T, gd)
        self.done = torch.zeros(T)
        self.ptr  = 0

    def add(self, **kw):
        for k, v in kw.items():
            getattr(self, k)[self.ptr] = v
        self.ptr += 1

    def data(self):
        sl = slice(0, self.ptr)
        return {k: getattr(self, k)[sl] for k in vars(self) if k not in {"T", "n", "ptr"}}

    def clear(self):
        self.ptr = 0

class Trainer:
    def __init__(self, env: CodeGenEnv, cfg: CFG):
        self.env, self.cfg = env, cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n, od, gd = env.n_agents, env.obs_dim, env.glob_dim
        self.lat_net = LatentNet(od, 64).to(self.device)
        self.actor   = LLMActor().to(self.device)
        self.crit    = MLP(gd, 1, hid=128, layers=3).to(self.device)
        self.inf     = InfNet(n, 3, gd).to(self.device)

        # optimizers
        self.optA = torch.optim.Adam(
            itertools.chain(
                self.lat_net.parameters(),
                self.actor.head.parameters(),
                self.actor.model.parameters(),
            ),
            lr=cfg.lr,
        )
        self.optC = torch.optim.Adam(self.crit.parameters(), lr=cfg.lr)
        self.optI = torch.optim.Adam(self.inf.parameters(),  lr=cfg.lr_inf)

        self.buf = Buffer(env.max_steps, n, od, gd)

    # ────────────────────────── rollout ────────────────────────────────────
    def collect(self) -> float:
        self.buf.clear()
        obs, glob = self.env.reset()
        mem = torch.zeros(self.env.n_agents, 64, device=self.device)

        for _ in range(self.env.max_steps):
            # same prompt for all agents (simple prototype)
            self.actor.set_prompts([self.env.task["prompt"]] * self.env.n_agents)

            o_t = torch.tensor(obs,  dtype=torch.float32, device=self.device)
            g_t = torch.tensor(glob, dtype=torch.float32, device=self.device)

            z, mu, sig = self.lat_net(o_t, mem)
            dist, _ = self.actor(o_t, mem, z)
            act = dist.sample()
            logp = dist.log_prob(act)

            val = self.crit(g_t).squeeze(-1).repeat(self.env.n_agents)

            obs, glob, rew, done, _ = self.env.step(act.cpu().numpy())
            self.buf.add(
                obs=o_t.cpu(), lat=z.cpu(), mu=mu.cpu(), sig=sig.cpu(),
                act=act.cpu(), logp=logp.cpu(), rew=torch.tensor(rew),
                val=val.cpu(), glob=g_t.cpu(), done=torch.tensor(done),
            )
            if done:
                break

        last_val = self.crit(torch.tensor(glob, dtype=torch.float32, device=self.device)).item()
        return last_val

    # ────────────────────────── GAE & returns ──────────────────────────────
    def adv_ret(self, data, last_v):
        T = data["rew"].shape[0]
        adv = torch.zeros_like(data["rew"])
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1 - data["done"][t].item()
            delta = (
                data["rew"][t]
                + self.cfg.gamma * (last_v if t == T - 1 else data["val"][t + 1]) * mask
                - data["val"][t]
            )
            gae = delta + self.cfg.gamma * self.cfg.lam * mask * gae
            adv[t] = gae
        return adv, adv + data["val"]

    # ────────────────────────── PPO update ────────────────────────────────
    def ppo_update(self, data, adv, ret):
        cfg, dev = self.cfg, self.device
        flat = lambda x: x.view(-1).to(dev)

        obs   = data["obs"].view(-1, self.env.obs_dim).to(dev)
        act   = flat(data["act"])
        oldlp = flat(data["logp"])
        adv_f = flat(adv)
        ret_f = flat(ret)
        g     = data["glob"].to(dev)
        mu    = data["mu"].to(dev)
        sig   = data["sig"].to(dev)
        lat   = data["lat"].to(dev)
        h0    = torch.zeros(len(obs), 64, device=dev)

        for _ in range(cfg.me):
            idx = torch.randperm(len(obs))
            for i in range(0, len(obs), cfg.bs):
                mb = idx[i : i + cfg.bs]

                self.actor.set_prompts([self.env.task["prompt"]] * len(mb))
                z, _, _ = self.lat_net(obs[mb], h0[mb])
                dist, _ = self.actor(obs[mb], h0[mb], z)
                lp = dist.log_prob(act[mb])

                ratio = (lp - oldlp[mb]).exp()
                surr = torch.min(
                    ratio * adv_f[mb],
                    torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_f[mb],
                ).mean()
                entropy = dist.entropy().mean()

                v = self.crit(g[mb]).repeat(self.env.n_agents)
                vloss = F.mse_loss(v, ret_f[mb])

                Le = (0.5 * math.log(2 * math.pi * math.e) + torch.log(sig[mb])).mean()
                Ld = cosine_dist(lat[mb])
                iloss = F.mse_loss(self.inf(g[mb], mu[mb], sig[mb]), ret_f[mb])

                loss = (
                    -surr
                    + cfg.ent * entropy
                    + cfg.vf * vloss
                    + cfg.le * Le
                    - cfg.ld * Ld
                    + cfg.lv * iloss
                )

                self.optA.zero_grad()
                self.optC.zero_grad()
                self.optI.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.lat_net.parameters(),
                        self.actor.head.parameters(),
                        self.actor.model.parameters(),
                    ),
                    cfg.max_g,
                )
                self.optA.step()
                self.optC.step()
                self.optI.step()

    # ────────────────────────── main loop ──────────────────────────────────
    def train(self, episodes: int):
        for ep in trange(episodes):
            last_v = self.collect()
            data = self.buf.data()
            adv, ret = self.adv_ret(data, last_v)
            self.ppo_update(data, adv, ret)

            total_r = data["rew"].sum().item()
            pf = self.env._pass_frac()

            if ep == 0:
                wandb.define_metric("episode")
                wandb.define_metric("total_reward", summary="max")

            table = wandb.Table(columns=["problem", "code", "pass_frac"])
            snippet = (self.env.code[:120] + "…") if self.env.code else "<empty>"
            table.add_data(self.env.task["name"], snippet, pf)
            wandb.log(
                {
                    "episode": ep,
                    "total_reward": total_r,
                    "pass_frac": pf,
                    "samples": table,
                }
            )

# ==============================================================================
# 6. CLI entry -----------------------------------------------------------------
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500,  help="training episodes")
    parser.add_argument("--split",    type=str, default="train",
                        choices=["train", "validation", "public"],
                        help="dataset split")
    args = parser.parse_args()

    wandb.init(project="marcog-codegen", config=vars(args))

    env = CodeGenEnv(split=args.split)
    trainer = Trainer(env, CFG())
    trainer.train(args.episodes)
