import logging
import wandb
import os
import sys
import transformers
from rich.logging import RichHandler
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from tqdm import trange

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from yy_env import (
    create_env,
    EnvConfig, 
    CurriculumConfig,
)

from shppo.model import build_actor_and_critic
from shppo.utils import RolloutBuffer, check_for_nan_and_abort

# ───────── Prompt repository keyed by agent role ─────────
ROLE_PROMPTS = {
    "planner": [
        {"role": "system", "content": (
            "You are the planner. Create a concise, step-by-step solution outline in markdown. "
            "Include function names, brief descriptions, and any edge cases to handle."
        )},
        {"role": "user", "content": (
            "{task_description}\n\nProvide the detailed plan."
        )},
    ],
    "coder": [
        {"role": "system", "content": (
            "You are the coder. Implement `solve(input_str: str) -> str` based on the planner's plan. "
        )},
        {"role": "user", "content": (
            "Plan:\n{planner_plan}\n\nWrite the `solve` function and runner."
        )},
    ],
    "debugger": [
        {"role": "system", "content": (
            "You are the debugger. Examine the `solve` function code and its execution output. "
            "Fix bugs, handle edge cases, and return the corrected function with brief explanations."
        )},
        {"role": "user", "content": (
            "Code:\n{code}\n\nOutput:\n{execution_output}\n\nProvide the corrected `solve` function."
        )},
    ],
}


@dataclass
class TeamResult:
    plan: str
    code: str
    execution_result: List[Dict]

# ══════════════════ 1. Configs ══════════════════
@dataclass
class RoleConfig:
    role_name: str
    obs_embed_dim: int

@dataclass
class SHPPOConfig:
    # Lora configuration
    base_model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_r: int = 8
    lora_alpha: int = 16
    
    # Model configuration
    latent_dim: int = 16
    hete_layer_input_dim: int = 512
    hete_layer_output_dim: int = 64
    mlp_hidden_dim: int = 256
    
    # agent configuration
    num_agents: int = 2
    role_configs: List[RoleConfig] = field(default_factory=lambda: [
        RoleConfig(role_name="planner", obs_embed_dim=64),
        RoleConfig(role_name="coder", obs_embed_dim=64)
    ])
    
    # Train configuration
    num_episodes: int = 1000
    lr = 1e-4
    warmup_ratio: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # runtime flags (copied from PPOConfig defaults)
    use_mixed_precision: bool = True
    cleanup_frequency: int = 10
    inf_batchsize: int = 32
    num_envs: int = 8
    train_batchsize: int = 32
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 0.5
    lambda_e: float = 0.01  # entropy regularization coefficient
    lamda_d: float = 0.01  # diversity regularization coefficient
    
    # Env configuration
    env_type: str = "curriculum_error"
    max_problem_length: int = 512
    max_solution_length: int = 512
    max_problems: int = 100_000
    
    def __post_init__(self):
        # there should be at least one coder
        if not any(role.role_name == "coder" for role in self.role_configs):
            raise ValueError("At least one role must be a coder.")
        
        # sort roles by planner -> coder -> debugger
        self.role_configs.sort(
            key=lambda x: ["planner", "coder", "debugger"].index(x.role_name) \
                if x.role_name in ["planner", "coder", "debugger"] else 3)

# =═══════════════════════════════════════════════════════════════════════════logger, 
# Set up rich logging
# ═══════════════════════════════════════════════════════════════════════
RICH_FORMAT = "[%(filename)s:%(lineno)s] >> %(message)s"

logging.basicConfig(
    level="INFO",
    format=RICH_FORMAT,
    handlers=[RichHandler(rich_tracebacks=True)],
)
# Set up rich logging
logger = logging.getLogger(__name__)
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.exit(0)
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
sys.excepthook = handle_exception

class SHPPOTrainer:
    def __init__(self, config: SHPPOConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        self.global_step = 0
        
        (
            self.actors,
            self.critic,
            self.latent_net,
            self.inference_net,
        ) = build_actor_and_critic(config, self.device)
        self.actors = [torch.compile(actor) for actor in self.actors]
        self.critic = torch.compile(self.critic)
        self.latent_net = torch.compile(self.latent_net).to(self.device)
        self.inference_net = torch.compile(self.inference_net).to(self.device)
        
        # optimizer: only use parameters from actors[0], critic, latent_net, inference_net
        self.optimizer = torch.optim.Adam(
            list(self.actors[0].parameters())
            + list(self.critic.parameters())
            + list(self.latent_net.parameters())
            + list(self.inference_net.parameters()),
            lr=config.lr, eps = 1e-8, weight_decay=1e-4
        )
        self.scheduler = transformers.get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(config.num_episodes * config.warmup_ratio),
            num_training_steps=config.num_episodes
        )
        
        self._setup_env()
        self.buffer = RolloutBuffer(config.gamma, config.gae_lambda)
        wandb.init(project="shppo", config=config, reinit=True)
        
        
    def _setup_env(self):
        train_env_cfg = EnvConfig(
            batch_size=self.cfg.inf_batchsize,
            max_problem_length=self.cfg.max_problem_length,
            max_solution_length=self.cfg.max_solution_length,
            max_problems=100_000,
            split="train",
        )
        eval_env_cfg = EnvConfig(
            batch_size=self.cfg.inf_batchsize,
            max_problem_length=self.cfg.max_problem_length,
            max_solution_length=self.cfg.max_solution_length,
            max_problems=100,
            split="valid",
        )
        test_env_cfg = EnvConfig(
            batch_size=self.cfg.inf_batchsize,
            max_problem_length=self.cfg.max_problem_length,
            max_solution_length=self.cfg.max_solution_length,
            max_problems=100_000,
            split="test",
        )
        
        if "curriculum" in self.cfg.env_type:
            self.curriculum_cfg = CurriculumConfig()
        else:
            self.curriculum_cfg = None
        
        self.env = create_env(self.cfg.env_type, train_env_cfg, self.curriculum_cfg)
        self.eval_env = create_env(self.cfg.env_type, eval_env_cfg, self.curriculum_cfg)
        self.test_env = create_env(self.cfg.env_type, test_env_cfg, self.curriculum_cfg)
        
        logger.info(f"🌟 Created environments with type: {self.cfg.env_type}")
        
        
    def collect_rollout(self):
        """
        Collect a multi‑agent rollout (planner → coder → optional debugger) and
        push it to the shared RolloutBuffer. Now runs for num_agents steps per episode,
        maintaining per-agent hidden states.
        """
        total_samples = self.cfg.num_envs
        batch_size    = self.cfg.inf_batchsize

        # Map each role to *all* matching actor indices (supports multiple planners, coders, etc.)
        role_to_indices: Dict[str, List[int]] = {}
        for i, r in enumerate(self.cfg.role_configs):
            role_to_indices.setdefault(r.role_name, []).append(i)

        planner_idxs  = role_to_indices.get("planner", [])
        coder_idxs    = role_to_indices.get("coder", [])
        debugger_idxs = role_to_indices.get("debugger", [])

        has_planner  = len(planner_idxs) > 0
        has_debugger = len(debugger_idxs) > 0

        # Use the first coder/debugger for now (extend later if needed)
        coder_actor    = self.actors[coder_idxs[0]] if coder_idxs else self.actors[0]
        planner_actors = [self.actors[i] for i in planner_idxs]
        debugger_actor = self.actors[debugger_idxs[0]] if has_debugger else None

        gpu_data_batches : list[dict] = []

        # Number of steps per episode = number of agents
        max_steps = self.cfg.num_agents

        # Initialize per-agent hidden states
        h_prevs = [None] * len(self.actors)

        for start in trange(0, total_samples, batch_size,
                            desc="GPU inference", position=1, leave=False):

            # Containers for per‑role log‑probs
            planner_logprobs_list: List[torch.Tensor] = []

            # Multi-step within this batch
            for step in range(max_steps):
                # Use persistent hidden states for each actor
                # ── 1. Get (or retain) the current batch of tasks ─────────────
                if step == 0:
                    self.env.reset_batch()
                tasks  = self.env.batch
                cur_bs = len(tasks)

                # Compute global feature: one-hot difficulty per task (easy, medium, hard, unknown)
                global_feats = []
                for t in tasks:
                    tags = set(t.get("cf_tags", []))
                    # Determine tag sets
                    easy_tags = set(getattr(self.cfg, "easy_tags", []))
                    medium_tags = set(getattr(self.cfg, "medium_tags", []))
                    hard_tags = set(getattr(self.cfg, "hard_tags", []))
                    # Classify
                    if not tags:
                        idx = 3  # unknown
                    elif tags & easy_tags:
                        idx = 0
                    elif tags & medium_tags:
                        idx = 1
                    elif tags & hard_tags:
                        idx = 2
                    else:
                        idx = 3  # unknown
                    onehot = [0] * 4
                    onehot[idx] = 1
                    global_feats.append(onehot)
                global_feats = torch.tensor(global_feats, dtype=torch.float32, device=self.device)

                # Track artefacts per task
                team_results = [TeamResult(plan="", code="", execution_result=[])
                                for _ in range(cur_bs)]

                # ── 2. Planner(s) (optional, may be multiple) ──────────────────
                if has_planner:
                    # For each task we accumulate every planner's output,
                    # then concatenate them before handing to the coder.
                    collected_plans: List[List[str]] = [[] for _ in range(cur_bs)]
                    # Save prompts for PPO
                    prompt_texts_list = []
                    for idx, pa in zip(planner_idxs, planner_actors):
                        planner_prompts = []
                        for t in tasks:
                            planner_prompts.append([
                                ROLE_PROMPTS["planner"][0],
                                {
                                    **ROLE_PROMPTS["planner"][1],
                                    "content": ROLE_PROMPTS["planner"][1]["content"].format(
                                        task_description=t["description"]
                                    ),
                                },
                            ])
                        # Save prompts for PPO
                        prompt_texts = []
                        for p in planner_prompts:
                            if hasattr(pa.llm.tokenizer, "apply_chat_template"):
                                prompt_texts.append(pa.llm.tokenizer.apply_chat_template(p, tokenize=False))
                            else:
                                prompt_texts.append("\n".join(f"{d['role']}: {d['content']}" for d in p))
                        prompt_texts_list = prompt_texts.copy()
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                plan_gen = pa.generate(planner_prompts, h_prev=h_prevs[idx])
                        h_prevs[idx] = plan_gen.get('h_next', None)
                        planner_logprobs_list.append(plan_gen["logprobs"])
                        plans = plan_gen["texts"]
                        for i in range(cur_bs):
                            collected_plans[i].append(plans[i])
                    # Concatenate all planners' outputs (separated by blank lines)
                    for i in range(cur_bs):
                        team_results[i].plan = "\n\n".join(collected_plans[i])

                # ── 3. Coder ────────────────────────────────────────────────────
                coder_prompts = []
                for i in range(cur_bs):
                    coder_prompts.append([
                        ROLE_PROMPTS["coder"][0],
                        {
                            **ROLE_PROMPTS["coder"][1],
                            "content": ROLE_PROMPTS["coder"][1]["content"].format(
                                planner_plan=team_results[i].plan
                            ),
                        },
                    ])
                # Save prompts for PPO
                coder_prompt_texts = []
                for p in coder_prompts:
                    if hasattr(coder_actor.llm.tokenizer, "apply_chat_template"):
                        coder_prompt_texts.append(coder_actor.llm.tokenizer.apply_chat_template(p, tokenize=False))
                    else:
                        coder_prompt_texts.append("\n".join(f"{d['role']}: {d['content']}" for d in p))
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        coder_gen = coder_actor.generate(coder_prompts, h_prev=h_prevs[coder_idxs[0]])
                h_prevs[coder_idxs[0]] = coder_gen.get('h_next', None)
                coder_sequences = coder_gen["sequences"]
                coder_logprobs  = coder_gen["logprobs"]
                codes           = coder_gen["texts"]
                mu              = coder_gen["mu"]
                sigma           = coder_gen["sigma"]
                for i, c in enumerate(codes):
                    team_results[i].code = c
                # First execution
                exec_rewards = self.env.step_batch(codes)
                for i, r in enumerate(exec_rewards):
                    team_results[i].execution_result.append(r)

                # ── 4. Debugger (optional) ──────────────────────────────────────
                final_seq     = coder_sequences
                final_logprob = coder_logprobs
                final_codes   = codes
                final_mu      = mu
                final_sigma   = sigma
                if has_debugger:
                    # Prepare debugger prompts using the coder's output
                    dbg_prompts = []
                    for i in range(cur_bs):
                        exec_lines = [
                            f"{r['input']}: {r['output']} {'SUCCESS' if r['succeed'] else 'FAIL'}"
                            for r in team_results[i].execution_result
                        ]
                        execution_output = "\n".join(exec_lines)
                        dbg_prompts.append([
                            ROLE_PROMPTS["debugger"][0],
                            {
                                **ROLE_PROMPTS["debugger"][1],
                                "content": ROLE_PROMPTS["debugger"][1]["content"].format(
                                    code=team_results[i].code,
                                    execution_output=execution_output,
                                ),
                            },
                        ])
                    # Save prompts for PPO
                    debugger_prompt_texts = []
                    for p in dbg_prompts:
                        if hasattr(debugger_actor.llm.tokenizer, "apply_chat_template"):
                            debugger_prompt_texts.append(debugger_actor.llm.tokenizer.apply_chat_template(p, tokenize=False))
                        else:
                            debugger_prompt_texts.append("\n".join(f"{d['role']}: {d['content']}" for d in p))
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            dbg_gen = debugger_actor.generate(dbg_prompts, h_prev=h_prevs[debugger_idxs[0]])
                    h_prevs[debugger_idxs[0]] = dbg_gen.get('h_next', None)
                    final_seq     = dbg_gen["sequences"]
                    final_logprob = dbg_gen["logprobs"]
                    final_codes   = dbg_gen["texts"]
                    final_mu      = dbg_gen["mu"]
                    final_sigma   = dbg_gen["sigma"]
                    dbg_rewards = self.env.step_batch(final_codes)
                    for i, r in enumerate(dbg_rewards):
                        team_results[i].execution_result.append(r)
                    rewards = dbg_rewards
                else:
                    rewards = exec_rewards

                # Consolidate planner log‑probs into a tensor of shape
                #   (num_planners, batch_size)  or  None if no planner
                planner_logprobs = (
                    torch.stack(planner_logprobs_list) if planner_logprobs_list else None
                )

                # ── 5. Critic forward & sanity checks ──────────────────────────
                with torch.no_grad():
                    if self.cfg.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = coder_actor.llm.model(
                                input_ids=final_seq,
                                output_hidden_states=True,
                            )
                    else:
                        outputs = coder_actor.llm.model(
                            input_ids=final_seq,
                            output_hidden_states=True,
                        )
                    last_hidden = outputs.hidden_states[-1][:, -1, :].float()
                    check_for_nan_and_abort(last_hidden, "hidden states", logger, f"batch start={start}")
                    # Centralized critic over all agents’ latents + global feature
                    # Collect mu/sigma for all agents
                    # If all_generations is available, collect from each agent
                    all_generations = []
                    if has_planner:
                        for idx, pa in zip(planner_idxs, planner_actors):
                            # For planner, need to get mu/sigma if available
                            # Not available in current code, so fallback to coder/debugger
                            pass
                    # For coder and debugger
                    mu_list = []
                    sigma_list = []
                    mu_list.append(mu)
                    sigma_list.append(sigma)
                    if has_debugger:
                        mu_list[-1] = final_mu
                        sigma_list[-1] = final_sigma
                    # If multi-agent, this should include all agents' mu/sigma
                    mus    = torch.stack(mu_list, dim=1)
                    sigmas = torch.stack(sigma_list, dim=1)
                    joint  = torch.cat([
                        mus.view(mus.shape[0], -1),
                        sigmas.view(sigmas.shape[0], -1),
                        global_feats
                    ], dim=-1)
                    values = self.critic(joint).squeeze(-1).detach()
                    check_for_nan_and_abort(values, "critic values", logger, f"batch start={start}")

                # ── 6. Move to CPU and store ────────────────────────────────────
                gpu_data_batches.append({
                    "planner_logprobs": planner_logprobs.cpu() if planner_logprobs is not None else None,
                    "coder_logprobs"  : coder_logprobs.cpu(),
                    "debugger_logprobs": final_logprob.cpu() if has_debugger else None,
                    "states"  : last_hidden.cpu(),
                    "actions" : final_seq.cpu(),
                    "values"  : values.cpu(),
                    "rewards" : torch.tensor(rewards, dtype=torch.float32),
                    "mu": mu.cpu(),
                    "sigma": sigma.cpu(),
                    "global_feats": global_feats.cpu(),
                    # Store per-agent prompts for PPO
                    "prompts": coder_prompt_texts,
                })

                if start % self.cfg.cleanup_frequency == 0:
                    torch.cuda.empty_cache()

        # ── 7. Aggregate & normalize rewards ───────────────────────────────
        all_rewards = torch.cat([b["rewards"] for b in gpu_data_batches], dim=0)
        all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

        # ── 8. Push into RolloutBuffer -------------------------------------
        # Ensure the buffer has per‑role log‑prob containers
        if not hasattr(self.buffer, "planner_logprobs"):
            self.buffer.planner_logprobs = []
        if not hasattr(self.buffer, "coder_logprobs"):
            self.buffer.coder_logprobs = []
        if not hasattr(self.buffer, "debugger_logprobs"):
            self.buffer.debugger_logprobs = []
        if not hasattr(self.buffer, "mu"):
            self.buffer.mu = []
        if not hasattr(self.buffer, "sigma"):
            self.buffer.sigma = []
        if not hasattr(self.buffer, "global_feats"):
            self.buffer.global_feats = []
        if not hasattr(self.buffer, "prompts"):
            self.buffer.prompts = []

        self.buffer.reset()
        ptr = 0
        for b in gpu_data_batches:
            bs = b["rewards"].size(0)
            self.buffer.states   += list(b["states"])
            self.buffer.actions  += list(b["actions"])
            self.buffer.values   += list(b["values"])
            self.buffer.rewards  += list(all_rewards[ptr:ptr+bs])
            ptr += bs

            # Per‑role log‑probs
            if b["planner_logprobs"] is not None:
                # planner_logprobs shape: (num_planners, bs)
                self.buffer.planner_logprobs += list(b["planner_logprobs"].transpose(0, 1))
            else:
                self.buffer.planner_logprobs += [torch.tensor([])] * bs  # placeholder

            self.buffer.coder_logprobs   += list(b["coder_logprobs"])
            if b["debugger_logprobs"] is not None:
                self.buffer.debugger_logprobs += list(b["debugger_logprobs"])
            else:
                self.buffer.debugger_logprobs += [torch.tensor([])] * bs
            self.buffer.mu         += list(b["mu"])
            self.buffer.sigma      += list(b["sigma"])
            self.buffer.global_feats += list(b["global_feats"])
            # Store prompts for PPO
            self.buffer.prompts += list(b["prompts"])

        self.last_rewards = all_rewards.to(self.device)
    
    def update(self):
        """
        SH-PPO update: PPO-style policy update for LoRA, critic update, then latent/inference update.
        Each phase (policy, critic, latent/inference) does its own backward, optimizer.step(), and zero_grad().
        """
        # 1. Compute returns & advantages
        self.buffer.compute_returns_advantages()
        returns    = torch.stack(self.buffer.returns).to(self.device)
        advantages = torch.stack(self.buffer.advantages).to(self.device)

        # 2. POLICY UPDATE (LLM LoRA weights) via PPO, per-agent sequential
        clip_eps = getattr(self.cfg, "clip_epsilon", 0.1)
        for actor in self.actors:
            for _ in range(getattr(self.cfg, "ppo_epochs", 1)):
                # gather old data
                old_logp = torch.stack(self.buffer.logprobs).to(self.device)
                sequences = torch.stack(self.buffer.actions).to(self.device)
                # recompute log-probs under current actor policy
                new_logp = actor.llm._logp_from_scores(self.buffer.prompts, sequences).to(self.device)
                ratio = (new_logp - old_logp).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                (policy_loss / self.cfg.grad_accumulation_steps).backward()
                # Log gradients for debugging
                total_norm = 0.0
                for p in self.actors[0].parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = total_norm ** 0.5
                wandb.log({"grad_norm_actor": grad_norm}, step=self.global_step)
            # After finishing epochs for this actor, step and zero_grad
            self.optimizer.step()
            self.optimizer.zero_grad()

        # 3. CRITIC UPDATE via MSE
        values_pred = self.critic(torch.stack(self.buffer.states).to(self.device)).squeeze(-1)
        critic_loss = F.mse_loss(values_pred, returns)
        (critic_loss / self.cfg.grad_accumulation_steps).backward()
        # Log gradients for debugging
        total_norm = 0.0
        for p in self.critic.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5
        wandb.log({"grad_norm_critic": grad_norm}, step=self.global_step)
        # Step critic update
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 4. LATENT + INFERENCE UPDATE (existing logic)
        # Reuse your existing latent losses code here:
        #    compute Lv, Le, Ld per minibatch, backward with grad accumulation.
        # For brevity, we call the existing multi-batch latent update:
        latent_metrics = self._latent_inference_update()

        # 5. Only step scheduler (no optimizer.step() here)
        self.scheduler.step()

        return {
            "policy_loss": policy_loss.item(),
            "critic_loss": critic_loss.item(),
            **latent_metrics
        }


    def _latent_inference_update(self):
        """
        Extracts the previous latent-update logic into a helper that returns metrics.
        Performs backward, optimizer.step(), and zero_grad() for latent/inference phase.
        """
        # 1. Gather full mu, sigma, global_feats from buffer
        mu_full    = torch.stack(self.buffer.mu).to(self.device)
        sigma_full = torch.stack(self.buffer.sigma).to(self.device)
        global_full = torch.stack(self.buffer.global_feats).to(self.device)
        returns    = torch.stack(self.buffer.returns).to(self.device)

        # 2. Setup multi-batch parameters
        batch_size    = self.cfg.train_batchsize
        total_samples = returns.size(0)
        num_batches   = (total_samples + batch_size - 1) // batch_size

        critic_losses = []
        latent_losses = []
        entropy_terms = []
        diversity_terms = []

        for i in range(num_batches):
            start = i * batch_size
            end   = min(start + batch_size, total_samples)
            mb_mu    = mu_full[start:end]
            mb_sigma = sigma_full[start:end]
            mb_ret   = returns[start:end]
            mb_global = global_full[start:end]

            # InferenceNet loss Lv
            v_pred = self.inference_net(mb_mu, mb_sigma, mb_global).squeeze(-1)
            Lv     = 0.5 * F.mse_loss(v_pred, mb_ret)

            # Entropy Le and diversity Ld
            Le = 0.5 * torch.log(2*torch.pi*torch.e*mb_sigma.pow(2)).mean()
            dist_mat = torch.cdist(mb_mu, mb_mu, p=2)
            Ld = -dist_mat.mean()

            # Total latent loss
            latent_loss = Lv + self.cfg.lambda_e * Le - self.cfg.lamda_d * Ld
            (latent_loss / num_batches).backward()
            # Log gradients for debugging
            total_norm_latent = 0.0
            for p in self.latent_net.parameters():
                if p.grad is not None:
                    total_norm_latent += p.grad.data.norm(2).item() ** 2
            grad_norm_latent = total_norm_latent ** 0.5
            wandb.log({"grad_norm_latent": grad_norm_latent}, step=self.global_step)
            total_norm_inf = 0.0
            for p in self.inference_net.parameters():
                if p.grad is not None:
                    total_norm_inf += p.grad.data.norm(2).item() ** 2
            grad_norm_inf = total_norm_inf ** 0.5
            wandb.log({"grad_norm_inference": grad_norm_inf}, step=self.global_step)

            # Step latent/inference update
            self.optimizer.step()
            self.optimizer.zero_grad()

            critic_losses.append(Lv.item())
            latent_losses.append(latent_loss.item())
            entropy_terms.append(Le.item())
            diversity_terms.append(Ld.item())

        return {
            "latent_loss": sum(latent_losses)/len(latent_losses),
            "entropy": sum(entropy_terms)/len(entropy_terms),
            "diversity": sum(diversity_terms)/len(diversity_terms),
        }
    
    def train(self):
        for ep in range(self.cfg.num_episodes):
            self.collect_rollout()
            metrics = self.update()
            wandb.log(metrics, step=self.global_step)
            self.global_step += 1

    def evaluate(self):
        rewards = []
        for _ in range(10):
            self.collect_rollout()
            rewards.append(self.last_rewards.mean().item())
        avg_reward = sum(rewards) / len(rewards)
        print(f"Eval average reward: {avg_reward}")
        return avg_reward
    
if __name__ == "__main__":
    config = SHPPOConfig()
    trainer = SHPPOTrainer(config)
    logger.info("SHPPO Trainer initialized successfully.")
    
    # Example usage
    trainer.train()
    trainer.evaluate()
    logger.info("Training and evaluation completed.")