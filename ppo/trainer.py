from __future__ import annotations
import os, sys, random
from dataclasses import dataclass, field
from typing import List
import math
import logging
from rich.logging import RichHandler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler  # Mixed precision for speed
from tqdm import trange, tqdm
import wandb
from transformers import get_cosine_schedule_with_warmup

# --------------------------------------------------------------------------- #
# Local imports
# --------------------------------------------------------------------------- #
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from env import (
    create_env,
    EnvConfig, 
    CurriculumConfig,
)

from ppo.model import build_actor_and_critic
from ppo.utils import RolloutBuffer, set_seed, check_for_nan_and_abort, SimpleCache

# =═══════════════════════════════════════════════════════════════════════════logger, 
# Set up rich logging
# ═══════════════════════════════════════════════════════════════════════
RICH_FORMAT = "%(message)s"

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

# ════════════════════════════════════════════════════════════════════════════
# 1. PPO Configuration with minimal optimizations
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class PPOConfig:
    # LLM / LoRA (unchanged)
    base_model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_r: int = 16
    lora_alpha: int = 32

    # Critic (unchanged)
    critic_hidden_dims: List[int] = field(default_factory=lambda: [])

    # PPO (unchanged)
    updates: int = 500
    rollout_len: int = 1
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_epsilon: float = 0.05
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    lr: float = 1e-6
    final_lr: float = 1e-7
    max_grad_norm: float = 0.5
    warmup_ratio: float = 0.01  # Warmup ratio for learning rate schedule
    
    # Multiple PPO epochs (unchanged)
    ppo_epochs: int = 1
    mini_batch_size: int = 16
    grad_acc: int = 32  # Gradient accumulation steps
    kl_coef: float = 1.0
    
    # Top-K KL divergence settings (unchanged)
    topk_for_kl: int = 1_000_000
    use_topk_kl: bool = False

    # Environment (unchanged)
    num_envs: int = 512
    seed: int = 42
    max_problem_length: int = 2048
    max_solution_length: int = 512
    env_type: str = "mixed_curriculum"  # "simple_error", "mixed_curriculum", etc.
    
    # NEW: Simple performance optimizations
    use_mixed_precision: bool = True     # Easy 30-50% speedup
    use_compile: bool = True             # PyTorch 2.0 compile for speed
    cleanup_frequency: int = 10          # Memory cleanup every N batches


def compute_topk_logprobs_for_sequences_optimized(model, tokenizer, prompts, sequences, k=50, device='cuda', use_amp=True):
    """
    Optimized version with mixed precision and memory management
    """
    model_device = next(model.parameters()).device
    sequences = sequences.to(model_device)
    
    # Tokenize prompts
    prompt_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model_device)
    
    prompt_length = prompt_inputs["input_ids"].shape[1]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attention_mask = (sequences != pad_token_id).float().to(model_device)
    
    # Mixed precision forward pass for speed
    with torch.no_grad():
        if use_amp:
            with autocast():
                outputs = model(
                    input_ids=sequences, 
                    attention_mask=attention_mask,
                    use_cache=False
                )
                logits = outputs.logits.float()
        else:
            outputs = model(
                input_ids=sequences, 
                attention_mask=attention_mask,
                use_cache=False
            )
            logits = outputs.logits.float()
    
    # Extract generated tokens (skip prompt)
    generated_tokens = sequences[:, prompt_length:]
    available_length = logits.size(1)
    
    if prompt_length >= available_length:
        # print(f"⚠️  Prompt too long ({prompt_length} >= {available_length}), returning zeros")
        logger.warning(f"⚠️  Prompt too long ({prompt_length} >= {available_length}), returning zeros")
    
    # Get logits for generated positions
    generated_logits = logits[:, prompt_length-1:-1, :]
    
    min_length = min(generated_tokens.size(1), generated_logits.size(1))
    if min_length == 0:
        # print("⚠️  No generated tokens, returning zeros")
        logger.warning("⚠️  No generated tokens, returning zeros")
        return torch.zeros(generated_tokens.size(0), device=model_device)
    
    generated_tokens = generated_tokens[:, :min_length]
    generated_logits = generated_logits[:, :min_length, :]
    
    # Optimized top-k computation
    batch_size, seq_len, vocab_size = generated_logits.shape
    topk_values, topk_indices = torch.topk(generated_logits, k, dim=-1)
    
    # Create mask for top-k positions
    mask = torch.zeros_like(generated_logits, dtype=torch.bool, device=model_device)
    mask.scatter_(-1, topk_indices, True)
    
    # Check if generated tokens are in top-k
    generated_in_topk = mask.gather(-1, generated_tokens.unsqueeze(-1)).squeeze(-1)
    
    # Apply mask to logits (set non-top-k to -inf)
    masked_logits = torch.where(mask, generated_logits, torch.tensor(-float('inf'), device=model_device))
    
    # Get log probabilities for top-k only
    log_probs = F.log_softmax(masked_logits, dim=-1)
    
    # Extract log probs for generated tokens
    token_log_probs = log_probs.gather(-1, generated_tokens.unsqueeze(-1)).squeeze(-1)
    
    # For tokens not in top-k, use smaller penalty (unchanged logic)
    topk_min_log_probs = F.log_softmax(topk_values, dim=-1).min(dim=-1)[0]
    small_penalty = 1.0
    fallback_log_probs = topk_min_log_probs - small_penalty
    
    token_log_probs = torch.where(
        generated_in_topk, 
        token_log_probs, 
        fallback_log_probs
    )
    
    # Apply attention mask to ignore padding
    if attention_mask is not None:
        gen_attention_mask = attention_mask[:, prompt_length:prompt_length+min_length]
        token_log_probs = token_log_probs * gen_attention_mask
        
    sequence_log_probs = token_log_probs.sum(dim=-1)
    return sequence_log_probs.to(device)


# ════════════════════════════════════════════════════════════════════════════
# 2. PPO Trainer with minimal optimizations
# ════════════════════════════════════════════════════════════════════════════
class PPOTrainer:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(cfg.seed)
        self.global_step = 0

        # print(f"🔧 Using Top-K KL divergence: k={cfg.topk_for_kl}, enabled={cfg.use_topk_kl}")
        logger.info(f"⚡ Performance optimizations: Mixed Precision={cfg.use_mixed_precision}, Compile={cfg.use_compile}")

        # Mixed precision setup
        self.scaler = GradScaler() if cfg.use_mixed_precision else None
        
        # Simple cache for tokenization
        self.tokenize_cache = SimpleCache(max_size=500)

        # ───── Models ─────────────────────────────────────────────────── #
        (
            self.actor,
            self.critic,
            actor_params,
            critic_params,
            self.tokenizer,
        ) = build_actor_and_critic(cfg, self.device)

        # PyTorch 2.0 compilation for speed
        if cfg.use_compile and hasattr(torch, 'compile'):
            logger.info("🚀 Compiling models with torch.compile for speed...")
            self.actor = torch.compile(self.actor)
            self.critic = torch.compile(self.critic)

        # Optimiser (unchanged)
        self.optimizer = torch.optim.AdamW(
            list(actor_params) + list(critic_params),
            lr=cfg.lr, 
            eps=1e-8,
            weight_decay=1e-4,
        )
        from transformers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(cfg.warmup_ratio * cfg.updates),
            num_training_steps=cfg.updates,
        )

        # ───── Environment Setup (unchanged) ─────────────────────────── #
        self._setup_environments()

        # Rollout buffer (unchanged)
        self.buffer = RolloutBuffer(cfg.gamma, cfg.gae_lambda)

        # Weights & Biases (unchanged)
        wandb.init(project="ppo", config=cfg, reinit=True)
        
        self.total_updates = 0

    def _setup_environments(self):
        """Setup environments (unchanged)"""
        train_env_cfg = EnvConfig(
            batch_size=50,
            max_problem_length=self.cfg.max_problem_length,
            max_solution_length=self.cfg.max_solution_length,
            max_problems=100_000,
            split="train",
        )
        
        eval_env_cfg = EnvConfig(
            batch_size=10,
            max_problem_length=self.cfg.max_problem_length,
            max_solution_length=self.cfg.max_solution_length,
            max_problems=1_000,
            split="valid",
        )
        
        test_env_cfg = EnvConfig(
            batch_size=10,
            max_problem_length=self.cfg.max_problem_length,
            max_solution_length=self.cfg.max_solution_length,
            max_problems=100_000,
            split="test",
        )

        curriculum_cfg = None
        if "curriculum" in self.cfg.env_type:
            curriculum_cfg = CurriculumConfig(
                pass_rate_threshold=0.1,
                avg_reward_threshold=0.5,
                min_episodes_per_level=20,
                eval_window_size=20,
            )

        logger.info(f"🌟 Creating environments with type: {self.cfg.env_type}")
        
        self.env = create_env(self.cfg.env_type, train_env_cfg, curriculum_cfg)
        self.eval_env = create_env("simple_error", eval_env_cfg)
        self.test_env = create_env("simple_error", test_env_cfg)
        
        logger.info(f"✅ Environments created successfully!")
        
    def collect_rollout(self) -> None:
        """Rollout collection with optimizations"""
        total_samples = self.cfg.num_envs
        batch_size = self.env.cfg.batch_size
        
        gpu_data_batches = []
        all_texts_for_execution = []
        all_tasks_for_execution = []

        for batch_idx, start in enumerate(trange(0, total_samples, batch_size, desc="GPU inference", position=1, leave=False)):
            end = min(start + batch_size, total_samples)
            current_size = end - start

            self.env.reset_batch()
            prompts = [
                f"Please implement a Python function named `solve(input_str)` that takes the problem input as `input_str` "
                f"and returns the correct output as a string.\n\nProblem:\n{task['description']}"
                for task in self.env.batch
            ]

            with torch.no_grad():
                # Mixed precision generation for speed
                if self.cfg.use_mixed_precision:
                    with autocast():
                        gen = self.actor.generate(prompts)
                else:
                    gen = self.actor.generate(prompts)
                
                # Check for NaN in generation
                check_for_nan_and_abort(gen["sequences"], "generated sequences", logger, f"batch {batch_idx}")
                check_for_nan_and_abort(gen["logprobs"], "generated logprobs", logger, f"batch {batch_idx}")
                
                # Mixed precision for hidden states
                if self.cfg.use_mixed_precision:
                    with autocast():
                        outputs = self.actor.model(
                            input_ids=gen["sequences"],
                            output_hidden_states=True,
                        )
                else:
                    outputs = self.actor.model(
                        input_ids=gen["sequences"],
                        output_hidden_states=True,
                    )
                
                last_hidden = outputs.hidden_states[-1][:, -1, :].float()
                
                # Check for NaN in hidden states
                check_for_nan_and_abort(last_hidden, "hidden states", logger, f"batch {batch_idx}")
                
                # Critic processing (unchanged logic)
                if hasattr(self.critic, 'module'):
                    critic_module = self.critic.module
                else:
                    critic_module = self.critic
                
                critic_device = next(critic_module.parameters()).device
                if critic_device != last_hidden.device:
                    last_hidden_for_critic = last_hidden.to(critic_device)
                    if self.cfg.use_mixed_precision:
                        with autocast():
                            values = critic_module(last_hidden_for_critic).detach()
                    else:
                        values = critic_module(last_hidden_for_critic).detach()
                    del last_hidden_for_critic
                else:
                    if self.cfg.use_mixed_precision:
                        with autocast():
                            values = critic_module(last_hidden).detach()
                    else:
                        values = critic_module(last_hidden).detach()
                
                # Check for NaN in values
                check_for_nan_and_abort(values, "critic values", logger, f"batch {batch_idx}")
                
                # Move to CPU (unchanged)
                states_cpu = last_hidden.cpu()
                actions_cpu = gen["sequences"].cpu()
                logprobs_cpu = gen["logprobs"].cpu()
                values_cpu = values.cpu()
                texts = gen["texts"].copy()
                
                # GPU memory cleanup (more frequent for optimization)
                del last_hidden, outputs, gen, values
                if batch_idx % self.cfg.cleanup_frequency == 0:
                    torch.cuda.empty_cache()
                
            # Final NaN check before storing
            check_for_nan_and_abort(states_cpu, "states_cpu", logger, f"batch {batch_idx}")
            check_for_nan_and_abort(actions_cpu, "actions_cpu", logger, f"batch {batch_idx}")
            check_for_nan_and_abort(logprobs_cpu, "logprobs_cpu", logger, f"batch {batch_idx}")
            check_for_nan_and_abort(values_cpu, "values_cpu", logger, f"batch {batch_idx}")
            
            gpu_data_batches.append({
                'prompts': prompts.copy(),
                'states': states_cpu,
                'actions': actions_cpu, 
                'logprobs': logprobs_cpu,
                'values': values_cpu,
                'batch_size': len(prompts),
                'texts': texts,
                'tasks': self.env.batch.copy()
            })
            
            all_texts_for_execution.extend(texts)
            all_tasks_for_execution.extend(self.env.batch.copy())
            del states_cpu, actions_cpu, logprobs_cpu, values_cpu, texts
            
        torch.cuda.empty_cache()

        if not gpu_data_batches or not all_texts_for_execution:
            logger.warning("❌ No valid batches collected, cannot proceed with rollout")
            self.last_rewards = torch.zeros(1, dtype=torch.float32, device=self.device)
            return

        # Rest of the function unchanged...
        original_batch = self.env.batch.copy() if hasattr(self.env, 'batch') else []
        self.env.batch = all_tasks_for_execution
        
        all_rewards = self.env.step_batch(all_texts_for_execution)
        
        self.env.batch = original_batch
        
        # Validate rewards
        if len(all_rewards) != len(all_texts_for_execution):
            logger.error(f"❌ Reward count mismatch: {len(all_rewards)} rewards for {len(all_texts_for_execution)} texts")
            sys.exit(1)
            
        all_rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
        all_rewards_tensor = (all_rewards_tensor - all_rewards_tensor.mean()) / (all_rewards_tensor.std() + 1e-8)
        
        # Check for NaN in rewards - ABORT if found
        if torch.isnan(all_rewards_tensor).any():
            logger.error((
                "❌ NaN detected in rewards from environment\n"
                f"   Rewards: {all_rewards}\n"
                "🚨 ABORTING PROCESS DUE TO NaN IN REWARDS"
            ))
            
            sys.exit(1)
        
        reward_offset = 0
        collected_rewards = []
        
        # Initialize buffer prompts if needed
        if not hasattr(self.buffer, 'prompts') or self.buffer.prompts is None:
            self.buffer.prompts = []
        
        for batch_idx, batch_data in enumerate(tqdm(gpu_data_batches, desc="Adding to buffer", position=1, leave=False)):
            batch_size = batch_data['batch_size']
            
            if reward_offset + batch_size > len(all_rewards_tensor):
                logger.warning(f"⚠️  Batch {batch_idx}: Not enough rewards remaining.")
                batch_size = len(all_rewards_tensor) - reward_offset
                if batch_size <= 0:
                    logger.warning(f"⚠️  Skipping batch {batch_idx}: no rewards remaining")
                    break
            
            batch_rewards = all_rewards_tensor[reward_offset:reward_offset + batch_size]
            reward_offset += batch_size
            
            if len(batch_rewards) == 0:
                logger.warning(f"⚠️  Empty batch_rewards for batch {batch_idx}, skipping")
                continue
            
            self.buffer.prompts.extend(batch_data['prompts'][:batch_size])
            
            for i in range(batch_size):
                if i >= len(batch_rewards):
                    logger.warning(f"⚠️  Index {i} out of bounds for batch_rewards of size {len(batch_rewards)}")
                    break
                    
                self.buffer.add(
                    state=batch_data['states'][i],
                    action=batch_data['actions'][i], 
                    logprob=batch_data['logprobs'][i],
                    reward=batch_rewards[i],
                    value=batch_data['values'][i]
                )
            
            collected_rewards.append(batch_rewards)
            del batch_data

        self.last_rewards = torch.cat(collected_rewards, dim=0).to(self.device)
        
    def update(self):
        """PPO update with gradient accumulation and detailed debugging messages"""
        self.buffer.compute_returns_advantages()

        # Check for NaN in advantages and returns - ABORT if found
        check_for_nan_and_abort(self.buffer.advantages, "buffer advantages", logger, "update start")
        check_for_nan_and_abort(self.buffer.returns, "buffer returns", logger,  "update start")

        # 3. Advantage normalization with debugging
        advantages = self.buffer.advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std()

        # Check advantage statistics for NaN
        if torch.isnan(adv_mean) or torch.isnan(adv_std):
            logger.error((
                f"❌ NaN in advantage statistics: mean={adv_mean}, std={adv_std}\n"
                "🚨 ABORTING PROCESS DUE TO NaN IN ADVANTAGE STATISTICS"
            ))
            sys.exit(1)

        # Standard normalization
        epsilon = 1e-8
        advantages_normalized = (advantages - adv_mean) / (adv_std + epsilon)
        # Check normalized advantages for NaN
        check_for_nan_and_abort(advantages_normalized, "normalized advantagelogger, s", "after normalization")
        advantages_normalized = torch.clamp(advantages_normalized, -5, 5) # clip the normalized advantages to avoid extreme values

        # 4. Data preparation
        total_samples = len(self.buffer.actions)

        all_prompts = self.buffer.prompts
        all_states = torch.stack(self.buffer.states, dim=0)
        all_actions = self._pad_and_stack_actions(self.buffer.actions)
        all_old_logprobs = torch.stack(self.buffer.logprobs, dim=0)
        all_returns = self.buffer.returns
        all_advantages = advantages_normalized

        # Check stacked tensors for NaN
        check_for_nan_and_abort(all_states, "stacked states",logger,  "before update loop")
        check_for_nan_and_abort(all_actions, "stacked actions",logger,  "before update loop")
        check_for_nan_and_abort(all_old_logprobs, "stacked old_logprobs",logger,  "before update loop")

        # 5. Training statistics initialization
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl_div = 0.0
        total_batches = 0
        early_stopped = False

        # Gradient accumulation setup
        self.optimizer.zero_grad(set_to_none=True)
        grad_accum_steps = cfg.grad_acc

        # Multiple epochs
        for epoch in trange(self.cfg.ppo_epochs, desc="PPO epochs", position=1, leave=False):
            indices = torch.randperm(total_samples)
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            epoch_kl_div = 0.0
            epoch_batches = 0

            for batch_idx, start_idx in enumerate(range(0, total_samples, self.cfg.mini_batch_size)):
                end_idx = min(start_idx + self.cfg.mini_batch_size, total_samples)
                mb_indices = indices[start_idx:end_idx]

                if len(mb_indices) == 0:
                    logger.warning(f"  ⚠️  Skipping empty batch {batch_idx}")
                    continue

                # Extract mini-batch
                mb_prompts = [all_prompts[i] for i in mb_indices]
                mb_states = all_states[mb_indices]
                mb_actions = all_actions[mb_indices]
                mb_old_logprobs = all_old_logprobs[mb_indices]
                mb_returns = all_returns[mb_indices]
                mb_advantages = all_advantages[mb_indices]

                # Check mini-batch for NaN
                check_for_nan_and_abort(mb_states, "mini-batch states", logger, f"epoch {epoch}, batch {batch_idx}")
                check_for_nan_and_abort(mb_actions, "mini-batch actions", logger, f"epoch {epoch}, batch {batch_idx}")
                check_for_nan_and_abort(mb_old_logprobs, "mini-batch old_logprobs", logger, f"epoch {epoch}, batch {batch_idx}")
                check_for_nan_and_abort(mb_returns, "mini-batch returns", logger, f"epoch {epoch}, batch {batch_idx}")
                check_for_nan_and_abort(mb_advantages, "mini-batch advantages", logger, f"epoch {epoch}, batch {batch_idx}")

                # Compute losses with debugging
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    policy_loss, value_loss, entropy, kl_div = self._compute_losses_topk_kl_optimized(
                        mb_prompts, mb_states, mb_actions, mb_old_logprobs,
                        mb_returns, mb_advantages
                    )

                # Check losses for NaN - ABORT if found
                if torch.isnan(policy_loss):
                    logger.error("❌ NaN in policy loss")
                    sys.exit(1)

                if torch.isnan(value_loss):
                    logger.error("❌ NaN in value loss")
                    sys.exit(1)

                if torch.isnan(entropy):
                    logger.error("❌ NaN in entropy")
                    sys.exit(1)

                # Compute total loss
                total_loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                    + self.cfg.kl_coef * kl_div
                )
                total_loss = torch.clamp(total_loss, min=-5, max=5)

                # Check total loss for NaN
                check_for_nan_and_abort(total_loss, "total_loss", logger, f"epoch {epoch}, batch {batch_idx}")

                # --- Gradient accumulation logic ---
                if self.scaler:
                    scaled_loss = self.scaler.scale(total_loss / grad_accum_steps)
                    scaled_loss.backward()
                else:
                    (total_loss / grad_accum_steps).backward()

                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1 == math.ceil(total_samples / self.cfg.mini_batch_size)):
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            list(self.actor.parameters()) + list(self.critic.parameters()),
                            self.cfg.max_grad_norm
                        )
                        if not torch.isfinite(grad_norm):
                            logger.warning(f"⚠️ Non-finite grad norm: {grad_norm} — skipping update")
                            # Reset scaler state to allow subsequent unscale calls
                            self.scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                            return
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            list(self.actor.parameters()) + list(self.critic.parameters()),
                            self.cfg.max_grad_norm
                        )
                        if not torch.isfinite(grad_norm):
                            logger.warning(f"⚠️ Non-finite grad norm: {grad_norm} — skipping update")
                            self.optimizer.zero_grad(set_to_none=True)
                            return
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # Accumulate statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl_div += kl_div
                total_batches += 1

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
                epoch_kl_div += kl_div
                epoch_batches += 1

                # More frequent memory cleanup
                if total_batches % self.cfg.cleanup_frequency == 0:
                    torch.cuda.empty_cache()
            if early_stopped:
                break

        # Update learning rate
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        new_lr = self.optimizer.param_groups[0]['lr']

        self.total_updates += 1

        # Final statistics
        if total_batches > 0:
            avg_policy_loss = total_policy_loss / total_batches
            avg_value_loss = total_value_loss / total_batches
            avg_entropy = total_entropy / total_batches
            avg_kl_div = total_kl_div / total_batches
        else:
            logger.warning("  ⚠️  No batches processed due to early stopping")
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            avg_entropy = 0.0
            avg_kl_div = 0.0

        self.buffer.reset()
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl_divergence': avg_kl_div,
            'early_stopped': early_stopped
        }

    def _pad_and_stack_actions(self, actions):
        """Pad actions to the same length and stack them (unchanged)"""
        if not actions:
            return torch.empty(0, 0, dtype=torch.long)
        
        max_length = max(action.size(0) for action in actions)
        
        padded_actions = []
        for action in actions:
            if action.size(0) < max_length:
                pad_length = max_length - action.size(0)
                padding = torch.full((pad_length,), 
                                   self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0, 
                                   dtype=action.dtype, 
                                   device=action.device)
                padded_action = torch.cat([action, padding], dim=0)
            else:
                padded_action = action
            padded_actions.append(padded_action)
        
        return torch.stack(padded_actions, dim=0)

    def _compute_losses_topk_kl_optimized(self, prompts, states, actions, old_logprobs, returns, advantages):
        """Compute PPO losses with optimizations"""
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # Check inputs for NaN
        check_for_nan_and_abort(states, "loss_computation_states",logger,  "loss computation start")
        check_for_nan_and_abort(actions.float(), "loss_computation_actions",logger,  "loss computation start")
        check_for_nan_and_abort(old_logprobs, "loss_computation_old_logprobs",logger,  "loss computation start")
        check_for_nan_and_abort(returns, "loss_computation_returns",logger,  "loss computation start")
        check_for_nan_and_abort(advantages, "loss_computation_advantages",logger,  "loss computation start")
        
        # Memory cleanup before heavy computation
        if self.total_updates % self.cfg.cleanup_frequency == 0:
            torch.cuda.empty_cache()
        
        try:
            if self.cfg.use_topk_kl:
                # Use optimized Top-K log probability computation
                model = self.actor.model.module if hasattr(self.actor.model, "module") else self.actor.model
                new_logprobs = compute_topk_logprobs_for_sequences_optimized(
                    model, self.tokenizer, prompts, actions, 
                    k=self.cfg.topk_for_kl, device=self.device,
                    use_amp=self.cfg.use_mixed_precision
                )
            else:
                # Use original method (may cause infinite KL)
                new_logprobs = self._compute_logprobs_for_sequences_original_optimized(prompts, actions)
                new_logprobs = new_logprobs.to(self.device)
                
            check_for_nan_and_abort(new_logprobs, "new_logprobs",logger,  "after logprob computation")
        except torch.cuda.OutOfMemoryError:
            print("❌ OOM during log probability computation")
            sys.exit(1)
        
        # Compute importance ratio with simple, stable clamping (unchanged logic)
        log_ratio = new_logprobs - old_logprobs
        check_for_nan_and_abort(log_ratio, "log_ratio",logger,  "importance ratio computation")
        
        log_ratio_clamped = torch.clamp(log_ratio, min=-5.0, max=5.0)
        ratio = torch.exp(log_ratio_clamped)
        check_for_nan_and_abort(ratio, "importance_ratilogger, o", "after exp")
        
        # Use the clamped log_ratio for KL divergence calculation
        kl_div = -log_ratio_clamped.mean().item()
        
        # PPO clipped objective (unchanged)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon
        )
        
        check_for_nan_and_abort(policy_loss_1, "policy_loss_1",logger,  "policy loss computation")
        check_for_nan_and_abort(policy_loss_2, "policy_loss_2",logger,  "policy loss computation")
        
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        check_for_nan_and_abort(policy_loss, "final_policy_loss",logger,  "policy loss final")
        
        # Value loss with mixed precision
        if self.cfg.use_mixed_precision:
            with autocast():
                current_values = self.critic(states).squeeze(-1)
                value_loss = 0.5 * ((current_values - returns) ** 2).mean()
        else:
            current_values = self.critic(states).squeeze(-1)
            value_loss = 0.5 * ((current_values - returns) ** 2).mean()
            
        check_for_nan_and_abort(current_values, "current_values",logger,  "critic forward pass")
        check_for_nan_and_abort(value_loss, "value_loss",logger,  "value loss computation")
        
        # Entropy (unchanged)
        entropy = -new_logprobs.var()
        check_for_nan_and_abort(entropy, "entropy", logger, "entropy computation")
        
        # KL divergence validation (unchanged)
        if math.isnan(kl_div):
            logger.error((
                f"❌ NaN in KL divergence: {kl_div}\n"
                "🚨 ABORTING PROCESS DUE TO NaN IN KL DIVERGENCE"
            ))
            sys.exit(1)
        
        if math.isinf(kl_div):
            logger.error((
                f"❌ Infinite KL divergence: {kl_div}\n"
                "   This shouldn't happen with clamped log ratios!\n"
                "🚨 ABORTING PROCESS DUE TO INFINITE KL"
            ))
            sys.exit(1)
        
        kl_div = abs(kl_div)
        
        # Cleanup after computation
        if self.total_updates % self.cfg.cleanup_frequency == 0:
            torch.cuda.empty_cache()
        
        return policy_loss, value_loss, entropy, kl_div

    def _compute_logprobs_for_sequences_original_optimized(self, prompts, sequences):
        """Optimized original log probability computation"""
        model = self.actor.model.module if hasattr(self.actor.model, "module") else self.actor.model
        
        batch_size = sequences.size(0)
        max_batch_size = 8
        
        if batch_size <= max_batch_size:
            return self._compute_logprobs_single_batch_original_optimized(prompts, sequences)
        
        all_log_probs = []
        for i in range(0, batch_size, max_batch_size):
            end_idx = min(i + max_batch_size, batch_size)
            batch_prompts = prompts[i:end_idx]
            batch_sequences = sequences[i:end_idx]
            
            batch_log_probs = self._compute_logprobs_single_batch_original_optimized(batch_prompts, batch_sequences)
            check_for_nan_and_abort(batch_log_probs, f"batch_log_probs_{logger, i}", "logprob computation")
            all_log_probs.append(batch_log_probs)
            
            # More frequent cleanup
            if i % (max_batch_size * 2) == 0:
                torch.cuda.empty_cache()
        
        result = torch.cat(all_log_probs, dim=0)
        check_for_nan_and_abort(result, "concatenated_log_probs",logger,  "logprob computation final")
        return result
    
    def _compute_logprobs_single_batch_original_optimized(self, prompts, sequences):
        """Optimized single batch log probability computation"""
        model = self.actor.model.module if hasattr(self.actor.model, "module") else self.actor.model
        device = next(model.parameters()).device
        
        sequences = sequences.to(device)
        check_for_nan_and_abort(sequences.float(), "input sequences",logger,  "logprob computation start")
        
        # Use cache for tokenization
        cache_key = str(hash(tuple(prompts)))
        prompt_inputs = self.tokenize_cache.get_or_compute(
            cache_key,
            lambda: self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_problem_length,
            )
        ).to(device)
        
        prompt_length = prompt_inputs["input_ids"].shape[1]
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        attention_mask = (sequences != pad_token_id).float().to(device)
        
        max_seq_length = 2048
        if sequences.size(1) > max_seq_length:
            logger.warning(f"⚠️  Truncating sequences from {sequences.size(1)} to {max_seq_length} tokens")
            sequences = sequences[:, :max_seq_length]
            attention_mask = attention_mask[:, :max_seq_length]
        
        with torch.no_grad():
            # Mixed precision forward pass
            if self.cfg.use_mixed_precision:
                with autocast():
                    outputs = model(
                        input_ids=sequences,
                        attention_mask=attention_mask,
                        use_cache=False
                    )
                    logits = outputs.logits.float()
                    check_for_nan_and_abort(logits, "model logits", logger, "forward pass")
                    
            else:
                # Non-mixed precision fallback
                outputs = model(
                    input_ids=sequences,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                logits = outputs.logits.float()
                check_for_nan_and_abort(logits, "model logits", logger, "forward pass")
                
        
        # Process in chunks (unchanged logic but with optimizations)
        chunk_size = 512
        sequence_length = logits.size(1)
        
        all_log_probs = []
        for start_idx in range(0, sequence_length, chunk_size):
            end_idx = min(start_idx + chunk_size, sequence_length)
            chunk_logits = logits[:, start_idx:end_idx, :]
            
            chunk_log_probs = F.log_softmax(chunk_logits, dim=-1)
            check_for_nan_and_abort(chunk_log_probs, f"chunk_log_probs_{start_idx}", logger, "log_softmax")
            all_log_probs.append(chunk_log_probs)
            
            del chunk_logits, chunk_log_probs
            # More frequent cleanup
            if start_idx % (chunk_size * 4) == 0:
                torch.cuda.empty_cache()
        
        log_probs = torch.cat(all_log_probs, dim=1)
        check_for_nan_and_abort(log_probs, "concatenated_log_probs", logger, "after chunking")
        del all_log_probs
        
        generated_tokens = sequences[:, prompt_length:].to(device)
        
        available_length = log_probs.size(1)
        if prompt_length >= available_length:
            logger.warning(f"⚠️  Prompt length {prompt_length} >= available length {available_length}, returning zero log probs")
            return torch.zeros(sequences.size(0), device=device)
        
        generated_logits = log_probs[:, prompt_length-1:available_length-1, :]
        
        min_length = min(generated_tokens.size(1), generated_logits.size(1))
        if min_length == 0:
            logger.warning("⚠️  No generated tokens to compute log probs for, returning zeros")
            return torch.zeros(sequences.size(0), device=device)
            
        generated_tokens = generated_tokens[:, :min_length]
        generated_logits = generated_logits[:, :min_length, :]
        
        gen_attention_mask = attention_mask[:, prompt_length:prompt_length+min_length] if attention_mask is not None else torch.ones_like(generated_tokens).float().to(device)
        
        token_log_probs = generated_logits.gather(2, generated_tokens.unsqueeze(-1)).squeeze(-1)
        check_for_nan_and_abort(token_log_probs, "token_log_probs", logger, "after gather")
        
        masked_token_log_probs = token_log_probs * gen_attention_mask
        sequence_log_probs = masked_token_log_probs.sum(dim=-1)
        
        check_for_nan_and_abort(sequence_log_probs, "final_sequence_log_probs",logger,  "logprob computation end")
        
        del log_probs, generated_logits, token_log_probs, masked_token_log_probs
        torch.cuda.empty_cache()
        
        return sequence_log_probs

    def train(self):
        """Training loop with performance optimizations"""
        for upd in trange(self.cfg.updates, desc="PPO updates", position=0, leave=True):
            self.collect_rollout()
            update_info = self.update()
            self.env.reset_batch()

            # Check final rewards for NaN
            check_for_nan_and_abort(self.last_rewards, "final_rewards", logger, f"update {upd}")

            pass_rate = float((self.last_rewards >= 0.99).sum().item() / self.last_rewards.size(0))
            error_rate = float((self.last_rewards < 0).sum().item() / self.last_rewards.size(0))
            avg_rewards = self.last_rewards.mean().item()
            positive_rate = float((self.last_rewards > 0).sum().item() / self.last_rewards.size(0))
            
            # Check computed metrics for NaN
            if math.isnan(pass_rate) or math.isnan(error_rate) or math.isnan(avg_rewards) or math.isnan(positive_rate):
                # print(f"❌ NaN in computed metrics at update {upd}")
                # print(f"   pass_rate: {pass_rate}, error_rate: {error_rate}")
                # print(f"   avg_rewards: {avg_rewards}, positive_rate: {positive_rate}")
                # print("🚨 ABORTING PROCESS DUE TO NaN IN METRICS")
                logger.error((
                    f"x❌ NaN in computed metrics at update {upd}\n"
                    f"  pass_rate: {pass_rate}, error_rate: {error_rate}\n"
                    f"  avg_rewards: {avg_rewards}, positive_rate: {positive_rate}\n"
                    "🚨 ABORTING PROCESS DUE TO NaN IN METRICS"
                ))
                    
                sys.exit(1)
            
            # Record performance for curriculum
            if hasattr(self.env, 'record_episode_performance'):
                self.env.record_episode_performance(avg_rewards, pass_rate)

            # Logging with performance metrics
            if update_info is None:
                logger.warning(f"⚠️  Update {upd} returned None, skipping logging")
                log_data = {
                    "train/step": self.global_step,
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "train/policy_loss": 0.0,
                    "train/value_loss": 0.0,
                    "train/entropy": 0.0,
                    "train/kl_divergence": 0.0,
                    "train/early_stopped": False,
                    "train/average_reward": avg_rewards,
                    "train/pass_rate": pass_rate,
                    "train/error_rate": error_rate,
                    "train/positive_rate": positive_rate,
                }
            else:
                log_data = {
                    "train/step": self.global_step,
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "train/policy_loss": update_info['policy_loss'],
                    "train/value_loss": update_info['value_loss'],
                    "train/entropy": update_info['entropy'],
                    "train/kl_divergence": update_info['kl_divergence'],
                    "train/early_stopped": update_info['early_stopped'],
                    "train/average_reward": avg_rewards,
                    "train/pass_rate": pass_rate,
                    "train/error_rate": error_rate,
                    "train/positive_rate": positive_rate,
                }

            # Check all log values for NaN
            for key, value in log_data.items():
                if isinstance(value, float) and math.isnan(value):
                    logger.error((
                        f"❌ NaN in log data: {key} = {value}\n"
                        "🚨 ABORTING PROCESS DUE TO NaN IN LOG DATA"
                    ))
                    sys.exit(1)

            # Add curriculum status if available
            if hasattr(self.env, 'get_curriculum_status'):
                curriculum_status = self.env.get_curriculum_status()
                for key, value in curriculum_status.items():
                    log_data[f"curriculum/{key}"] = value

            # Add performance stats
            env_stats = self.env.get_performance_stats()
            for key, value in env_stats.items():
                log_data[f"env_stats/{key}"] = value

            wandb.log(log_data, step=self.global_step)

            self.global_step += 1
            
            if upd % 10 == 0:
                self.evaluate(100)

    @torch.no_grad()
    def _batch_eval(self, env, tasks: list[dict]) -> tuple[float, float]:
        """Evaluation with optimizations"""
        batch_size = 100
        total_passed = 0
        total_success = 0
        total = 0
        
        for i in range(0, len(tasks), batch_size):
            try:
                batch = tasks[i:i+batch_size]
                env.batch = batch
                prompts = [
                    f"Please implement a Python function named `solve(input_str)` that takes the problem input as `input_str` "
                    f"and returns the correct output as a string.\n\nProblem:\n{task['description']}"
                    for task in batch
                ]
                
                # Mixed precision generation for evaluation
                if self.cfg.use_mixed_precision:
                    with autocast():
                        gen = self.actor.generate(prompts)
                else:
                    gen = self.actor.generate(prompts)
                
                # Check for NaN in evaluation generation
                check_for_nan_and_abort(gen["logprobs"], "eval_generation_logprobs", logger, f"evaluation batch {i}")
                    
                rewards = env.step_batch(gen["texts"])
                
                # Check for NaN in evaluation rewards
                for j, r in enumerate(rewards):
                    if isinstance(r, float) and math.isnan(r):
                        # print(f"❌ NaN in evaluation reward at batch {i}, sample {j}: {r}")
                        # print("🚨 ABORTING PROCESS DUE TO NaN IN EVALUATION REWARDS")
                        logger.error((
                            f"❌ NaN in evaluation reward at batch {i}, sample {j}: {r}\n"
                            "🚨 ABORTING PROCESS DUE TO NaN IN EVALUATION REWARDS"
                        ))
                        sys.exit(1)
                
                total_passed += sum(r >= 0.0 for r in rewards)
                total_success += sum(r >= 0.99 for r in rewards)
                total += len(rewards)
                
            except Exception as e:
                print(f"❌ Error in evaluation batch {i}: {e}")
                continue
                
        if total == 0:
            return 0.0, 0.0
            
        avg_passed = total_passed / total
        pass_rate = total_success / total
        
        # Check final evaluation results for NaN
        if math.isnan(avg_passed) or math.isnan(pass_rate):
            # print(f"❌ NaN in evaluation results: avg_reward={avg_reward}, pass_rate={pass_rate}")
            # print("🚨 ABORTING PROCESS DUE TO NaN IN EVALUATION RESULTS")
            logger.error((
                f"❌ NaN in evaluation results: avg_reward={avg_passed}, pass_rate={pass_rate}\n"
                "🚨 ABORTING PROCESS DUE TO NaN IN EVALUATION RESULTS"
            ))
            sys.exit(1)
            
        return avg_passed, pass_rate

    def evaluate(self, n: int = 100):
        """Evaluation with optimizations"""
        try:
            all_probs = self.eval_env.get_all_problems()
            sample = random.sample(all_probs, min(n, len(all_probs)))
            avg_p, pass_r = self._batch_eval(self.eval_env, sample)
            
            wandb.log({
                "eval/average_passed": avg_p,
                "eval/pass_rate": pass_r,
                "eval/num_samples": len(sample),
            }, step=self.global_step)
            
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")

    def test(self):
        """Test with optimizations"""
        try:
            avg_p, pass_r = self._batch_eval(
                self.test_env, self.test_env.get_all_problems()
            )
            
            wandb.log({
                "test/average_passed": avg_p,
                "test/pass_rate": pass_r,
                "test/num_samples": len(self.test_env.get_all_problems()),
            }, step=self.global_step)
            
        except Exception as e:
            print(f"❌ Error in testing: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", type=str, default="curriculum_error",
                       choices=["simple_simple", "simple_error", "curriculum_simple", "curriculum_error"],
                       help="Environment type to use")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override PPO epochs")
    parser.add_argument("--updates", type=int, default=None,
                       help="Override number of updates")
    parser.add_argument("--lr", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--topk_kl", type=int, default=50,
                       help="Number of top tokens for KL computation")
    parser.add_argument("--disable_topk_kl", action="store_true",
                       help="Disable top-K KL divergence (use original method)")
    
    # Performance optimization arguments
    parser.add_argument("--disable_mixed_precision", action="store_true",
                       help="Disable mixed precision training")
    parser.add_argument("--disable_compile", action="store_true",
                       help="Disable torch.compile optimization")
    parser.add_argument("--cleanup_freq", type=int, default=10,
                       help="Memory cleanup frequency")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting optimized PPO training")
    logger.info(f"📊 Environment type: {args.env_type}")
    logger.info(f"🔧 Top-K KL: k={args.topk_kl}, enabled={not args.disable_topk_kl}")
    
    # Create configuration
    cfg = PPOConfig()
    cfg.env_type = args.env_type
    cfg.topk_for_kl = args.topk_kl
    cfg.use_topk_kl = not args.disable_topk_kl
    cfg.use_mixed_precision = not args.disable_mixed_precision
    cfg.use_compile = not args.disable_compile
    cfg.cleanup_frequency = args.cleanup_freq
    
    if args.epochs is not None:
        cfg.ppo_epochs = args.epochs
    if args.updates is not None:
        cfg.updates = args.updates
    if args.lr is not None:
        cfg.lr = args.lr
    
    logger.info((f"🔧 Settings:\n"
        f"  Environment: {cfg.env_type}\n"
        f"  PPO epochs: {cfg.ppo_epochs}\n"
        f"  Updates: {cfg.updates}\n"
        f"  Learning rate: {cfg.lr}\n"
        f"  Top-K for KL: {cfg.topk_for_kl}\n"
        f"  Use Top-K KL: {cfg.use_topk_kl}\n"
        f"  Mixed Precision: {cfg.use_mixed_precision}\n"
        f"  Torch Compile: {cfg.use_compile}\n"
        f"  Cleanup Frequency: {cfg.cleanup_frequency}"
    ))
    
    # Create and run trainer
    trainer = PPOTrainer(cfg)
    trainer.train()
    trainer.test()
    
    logger.info(f"✅ Training completed successfully!")
    logger.info(f"📊 Final statistics: {trainer.total_updates} updates completed")