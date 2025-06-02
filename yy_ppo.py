#!/usr/bin/env python3
"""
A full PPO training script using Qwen-LoRA as actor and a custom Critic,
with CodeTester (from test.py) as the environment. Single‐step episodes,
vectorized across multiple CodeTester instances.

Improvements:
- Adaptive learning rate with cosine/linear schedule
- GAE advantage normalization
- Value loss clipping
"""

import os
import math
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import random
import pandas as pd
import numpy as np
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import get_peft_model, LoraConfig

from test import CodeContestDataset, CodeTester

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class PPOConfig:
    # Environment / rollout
    num_envs: int = 6
    rollout_length: int = 1      # FIXED: 단일 action이므로, 1 고정
    gamma: float = 0.99
    lam: float = 0.95
    updates: int = 30  # Number of PPO updates

    # PPO update
    epochs: int = 80
    clip_eps: float = 0.1
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    max_grad_norm: float = 0.3
    batch_size: int = 6         # minibatch size for PPO (<= num_envs * rollout_length)

    # Adaptive Learning Rate
    init_lr: float = 1e-4       # Initial learning rate
    final_lr: float = 5e-6      # Final learning rate for decay
    lr_schedule: str = "linear" # "cosine", "linear", "constant"
    warmup_updates: int = 8     # Number of warmup updates

    # Advanced PPO Features
    use_gae_normalization: bool = True  # Normalize GAE advantages
    value_loss_clipping: bool = True    # Clip value loss
    value_clip_range: float = 0.1       # Value clipping range

    # Training
    max_problems_per_update: int = 6    # Number of problems to sample per update
    sample_space: int = 4               # Action space size (number of test cases per problem)
    
    # Model Configuration
    lora_r: int = 8
    lora_alpha: int = 16
    
    version: str = "v2.0"      # Version for logging

    def __post_init__(self):
        assert self.num_envs * self.rollout_length >= self.batch_size, \
            "Batch size must be less than or equal to num_envs * rollout_length, " \
            f"got {self.batch_size} > {self.num_envs * self.rollout_length}"
            
    def get_learning_rate(self, current_update: int) -> float:
        """Compute learning rate based on current update with warmup and decay"""
        if current_update < self.warmup_updates:
            # Linear warmup
            return self.init_lr * (current_update / self.warmup_updates)
        
        # Progress after warmup
        progress = (current_update - self.warmup_updates) / (self.updates - self.warmup_updates)
        progress = min(1.0, max(0.0, progress))
        
        if self.lr_schedule == "cosine":
            # Cosine annealing
            return self.final_lr + 0.5 * (self.init_lr - self.final_lr) * (1 + math.cos(math.pi * progress))
        elif self.lr_schedule == "linear":
            # Linear decay
            return self.init_lr - progress * (self.init_lr - self.final_lr)
        elif self.lr_schedule == "constant":
            return self.init_lr
        else:
            raise ValueError(f"Unknown learning rate schedule: {self.lr_schedule}")

# -----------------------------------------------------------------------------
# Critic Network
# -----------------------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,)


# -----------------------------------------------------------------------------
# Actor Setup
# -----------------------------------------------------------------------------
def build_actor_and_critic(cfg: PPOConfig, device: torch.device) -> tuple:
    # 1) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Actor base: Qwen in 4-bit NF4
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    actor_base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,   
    )
    actor_base.config.use_sliding_window_attention = False
    actor_base.eval()

    # 3) LoRA injection
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    actor = get_peft_model(actor_base, lora_cfg)
    actor.train()

    # 4) Critic: uses same hidden size as Qwen
    hidden_dim = actor_base.config.hidden_size
    critic = Critic(hidden_dim).to(device)

    # 5) Parameter lists - only LoRA parameters for actor
    actor_vars = [p for n, p in actor.named_parameters() if "lora_" in n and p.requires_grad]
    critic_vars = list(critic.parameters())

    print(f"Actor trainable parameters: {len(actor_vars)}")
    print(f"Critic parameters: {len(critic_vars)}")

    return actor, critic, actor_vars, critic_vars, tokenizer


# -----------------------------------------------------------------------------
# PPO Trainer with CodeTester Integration
# -----------------------------------------------------------------------------
class PPOTrainer:
    def __init__(self, cfg: PPOConfig, device: torch.device, log_file: str = None):
        
        self.cfg = cfg
        self.device = device
        self.log_file = log_file

        # Build actor, critic, tokenizer
        self.actor, self.critic, a_vars, c_vars, self.tokenizer = build_actor_and_critic(cfg, device)
        
        # Initialize optimizer with initial learning rate
        self.optimizer = optim.AdamW(
            a_vars + c_vars,
            lr=cfg.init_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        # Datasets and CodeTesters for train, validation, and test splits
        self.dataset = CodeContestDataset(split="train", max_cases=3)
        self.eval_dataset = CodeContestDataset(split="valid")
        self.test_dataset = CodeContestDataset(split="test")

        self.code_tester = CodeTester(
            dataset=self.dataset,
            batch_size=100,
            max_workers=4
        )
        self.eval_tester = CodeTester(
            dataset=self.eval_dataset,
            batch_size=100,
            max_workers=4
        )
        self.test_tester = CodeTester(
            dataset=self.test_dataset,
            batch_size=100,
            max_workers=4
        )
        
        print(f"Loaded {len(self.dataset.get_all_tasks())} tasks for training")

    def update_learning_rate(self, current_update: int):
        """Update learning rate based on schedule"""
        new_lr = self.cfg.get_learning_rate(current_update)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def sample_problems(self, num_problems: int) -> List[Dict[str, Any]]:
        """Sample random problems from the dataset"""
        all_tasks = self.dataset.get_all_tasks()
        return random.sample(all_tasks, min(num_problems, len(all_tasks)))

    def evaluate_solutions_with_tester(self) -> List[float]:
        """Evaluate solutions using the test cases, aggregate rewards per problem, and return per-problem rewards"""
        csv_file = self.log_file
        results = pd.read_csv(csv_file)
        os.remove(csv_file)  # Clean up after evaluation
        rewards = []
        # Group by problem_name
        grouped = results.groupby('problem_name')
        for _, group in grouped:
            case_rewards = []
            for _, row in group.iterrows():
                if row['passed']:
                    case_rewards.append(1.0)
                elif isinstance(row['execution_error'], str) and row['execution_error'] != "":
                    case_rewards.append(-1.0)
                else:
                    case_rewards.append(0.0)
            avg_reward = sum(case_rewards) / len(case_rewards) if case_rewards else 0.0
            rewards.append(avg_reward)
        return rewards

    def collect_rollouts(self) -> Dict[str, torch.Tensor]:
        """Collect rollouts by sampling problems and generating solutions"""
        # Sample problems for this rollout
        problems = self.sample_problems(self.cfg.num_envs)

        # Create a temporary dataset with the sampled problems
        temp_dataset = CodeContestDataset.__new__(CodeContestDataset)
        temp_dataset.tasks = problems

        # Create a temporary CodeTester with the sampled problems
        temp_tester = CodeTester(
            dataset=temp_dataset,
            batch_size=self.cfg.num_envs * self.cfg.sample_space,
            max_workers=4,
            log_file=self.log_file
        )

        # Use tester.run to generate and write CSV, also get responses & hidden/logprobs
        avg_rate, solutions, hidden_states, generated_tokens, log_probs = temp_tester.run(
            self.tokenizer,
            self.actor,
            return_hidden=True,
            return_logprobs=True,
            num_return_seqs=self.cfg.sample_space
        )

        # Sample one candidate per prompt using the policy over candidates
        batch_size = self.cfg.num_envs
        num_return_sequences = self.cfg.sample_space  # must match call above
        # Group generated solutions per prompt
        grouped_solutions = [
            solutions[i:i+num_return_sequences]
            for i in range(0, len(solutions), num_return_sequences)
        ]
        # Reshape tokens and log_probs into [batch_size, num_return_sequences, ...]
        tokens = generated_tokens.view(batch_size, num_return_sequences, -1)
        probs = log_probs.view(batch_size, num_return_sequences)
        # Sanitize log_probs to avoid NaN or -inf logits in Categorical
        probs = torch.nan_to_num(probs, nan=-1e4, neginf=-1e4)
        # Build categorical distribution and sample
        dist = Categorical(logits=probs)           # shape: [batch_size, num_return_sequences]
        chosen_indices = dist.sample()              # shape: [batch_size]
        # Select the sampled solutions and tokens
        solutions = [
            grouped_solutions[i][chosen_indices[i].item()]
            for i in range(batch_size)
        ]
        generated_tokens = torch.stack([
            tokens[i, chosen_indices[i]]
            for i in range(batch_size)
        ], dim=0)
        # Use sampled log-prob for PPO update
        log_probs = dist.log_prob(chosen_indices).to(self.device)
        # hidden_states shape: (batch_size, hidden_dim); no reshaping needed

        # Evaluate solutions to get rewards
        rewards = self.evaluate_solutions_with_tester()

        # Convert to tensors and move to device
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        hidden_states = hidden_states.to(self.device).float()
        log_probs = log_probs.to(self.device)

        # Get value estimates from critic
        with torch.no_grad():
            values = self.critic(hidden_states)

        return {
            "rewards": rewards_tensor,
            "values": values,
            "hidden_states": hidden_states,
            "log_probs": log_probs,
            "solutions": solutions,
            "prompts": [p['prompt'] for p in problems],
        }

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute advantages using Generalized Advantage Estimation (GAE) with optional normalization"""
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * gae
            advantages[t] = gae
        
        # GAE Normalization (improved stability)
        if self.cfg.use_gae_normalization and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages

    def ppo_update(self, batch: Dict[str, torch.Tensor]):
        """Perform PPO update with improved value loss and advantage computation"""
        rewards = batch["rewards"]
        old_values = batch["values"]
        hidden_states = batch["hidden_states"]
        old_log_probs = batch["log_probs"]
        
        # Compute advantages and returns
        advantages = self.compute_advantages(rewards, old_values)
        returns = rewards  # For single-step episodes, returns = rewards
        
        # Multiple PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        
        for epoch in range(self.cfg.epochs):
            # Get current value estimates
            current_values = self.critic(hidden_states)
            
            # Value loss with optional clipping
            if self.cfg.value_loss_clipping:
                # Clipped value loss (similar to policy clipping)
                value_pred_clipped = old_values + torch.clamp(
                    current_values - old_values, 
                    -self.cfg.value_clip_range, 
                    self.cfg.value_clip_range
                )
                value_losses = (current_values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = nn.MSELoss()(current_values, returns)
            
            # PPO policy loss with clipping
            # Recompute log_probs under current policy for the chosen solutions
            input_encodings = self.tokenizer(batch["solutions"], return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.actor(**input_encodings, return_dict=True)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

            # Prepare next-token labels
            labels = input_encodings.input_ids
            shift_logits = logits[:, :-1, :].contiguous()      # (batch, seq_len-1, vocab_size)
            shift_labels = labels[:, 1:].contiguous()           # (batch, seq_len-1)

            # Compute per-token log-probs
            log_probs_flat = F.log_softmax(shift_logits, dim=-1)       # (batch, seq_len-1, vocab_size)
            token_log_probs = log_probs_flat.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len-1)

            # Sum per-token log-probs to get sequence log_prob
            new_log_probs = token_log_probs.sum(dim=1)  # (batch,)

            # Compute ratio between new and old log_probs
            ratios = (new_log_probs - old_log_probs).exp()
            clipped_ratios = torch.clamp(ratios, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
            
            # Enhanced entropy calculation (using actual token probabilities)
            token_probs = F.softmax(shift_logits, dim=-1)
            entropy = -(token_probs * F.log_softmax(shift_logits, dim=-1)).sum(dim=-1).mean()
            entropy_loss = -entropy * self.cfg.ent_coef
            
            # Total loss
            loss = policy_loss + self.cfg.vf_coef * value_loss - entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.actor.parameters() if p.requires_grad] + list(self.critic.parameters()),
                self.cfg.max_grad_norm
            )
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        avg_policy_loss = total_policy_loss / self.cfg.epochs
        avg_value_loss = total_value_loss / self.cfg.epochs
        avg_entropy_loss = total_entropy_loss / self.cfg.epochs
        
        return avg_policy_loss, avg_value_loss, avg_entropy_loss

    def train(self, updates: int):
        wandb.init(
            project="PPO-CodeTester",
            config=self.cfg,
            name=f"PPO-Training-{self.cfg.version}-{updates}Updates",
            reinit=True
        )
        wandb.config.update(
            {
                "num_actor_params": sum(p.numel() for p in self.actor.parameters() if p.requires_grad),
                "num_critic_params": sum(p.numel() for p in self.critic.parameters() if p.requires_grad),
            }
        )
        generation_table = wandb.Table(
            columns=["Update", "Problem", "Generated Code", "Execution Result"]
        )
        
        """Main training loop"""
        print("Starting PPO training with improvements...")
        print(f"- Adaptive learning rate: {self.cfg.lr_schedule}")
        print(f"- GAE normalization: {self.cfg.use_gae_normalization}")
        print(f"- Value loss clipping: {self.cfg.value_loss_clipping}")
        
        best_avg_reward = 0.0
        best_val_pass_rate = 0.0
        
        for update in range(1, updates + 1):
            # Update learning rate
            current_lr = self.update_learning_rate(update - 1)  # 0-indexed for update
            
            # Collect rollouts
            batch = self.collect_rollouts()
            
            # Log to wandb
            for i in range(len(batch["prompts"])):
                generation_table.add_row(
                    update,
                    str(batch["prompts"][i])[:200] + "..." if len(str(batch["prompts"][i])) > 200 else str(batch["prompts"][i]),
                    str(batch["solutions"][i])[:300] + "..." if len(str(batch["solutions"][i])) > 300 else str(batch["solutions"][i]),
                    batch["rewards"][i].item()
                )
            
            # PPO update
            policy_loss, value_loss, entropy_loss = self.ppo_update(batch)
            
            # Compute metrics
            avg_reward = batch["rewards"].mean().item()
            max_reward = batch["rewards"].max().item()
            min_reward = batch["rewards"].min().item()
            avg_log_prob = batch["log_probs"].mean().item()
            
            # Track best performance
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                print(f"New best average reward: {best_avg_reward:.4f}")
            
            # Logging
            if update % 5 == 0:
                print(f"[Update {update:04d}] "
                      f"lr={current_lr:.2e}, "
                      f"avg_reward={avg_reward:.4f}, "
                      f"max_reward={max_reward:.4f}, "
                      f"min_reward={min_reward:.4f}, "
                      f"policy_loss={policy_loss:.4f}, "
                      f"value_loss={value_loss:.4f}, "
                      f"entropy_loss={entropy_loss:.4f}, "
                      f"avg_log_prob={avg_log_prob:.4f}")
            
            # Validation evaluation every 5 updates
            if update % 5 == 0:
                print(f"\n--- Validation Evaluation at Update {update} ---")
                try:
                    val_pass_rate = self.eval_tester.run(self.tokenizer, self.actor)
                    print(f"Validation pass rate: {val_pass_rate:.2f}%")
                    
                    # Track best validation pass rate
                    if val_pass_rate > best_val_pass_rate:
                        best_val_pass_rate = val_pass_rate
                        print(f"New best validation pass rate: {best_val_pass_rate:.2f}%")
                        
                except Exception as e:
                    print(f"Validation evaluation failed: {e}")
                    val_pass_rate = 0.0
                    
                print("--- End Validation Evaluation ---\n")
            else:
                val_pass_rate = 0.0  # No validation this update
            
            # Log metrics to Wandb
            wandb.log({
                "update": update,
                "train/learning_rate": current_lr,
                "train/avg_reward": avg_reward,
                "train/max_reward": max_reward,
                "train/min_reward": min_reward,
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "train/entropy_loss": entropy_loss,
                "train/avg_log_prob": avg_log_prob,
                "val/pass_rate": val_pass_rate if val_pass_rate > 0 else None,
                "val/best_pass_rate": best_val_pass_rate
            }, step=update)
            
            wandb.log({"generation": generation_table})
        
        # Final test split evaluation
        print("\n--- Final Test Evaluation ---")
        try:
            test_pass_rate = self.test_tester.run(self.tokenizer, self.actor)
            print(f"Test pass rate: {test_pass_rate:.2f}%")
            
            # Log final test pass rate
            wandb.log({
                "test/pass_rate": test_pass_rate
            })
        except Exception as e:
            print(f"Final test evaluation failed: {e}")
            test_pass_rate = 0.0
        
        print(f"Training complete!")
        print(f"Best average reward: {best_avg_reward:.4f}")
        print(f"Best validation pass rate: {best_val_pass_rate:.2f}%")
        print(f"Final test pass rate: {test_pass_rate:.2f}%")

    def full_evaluation(self):
        """Run a full evaluation using CodeTester.run() on validation and test splits"""
        print("Running full evaluation on validation split...")
        try:
            val_rate = self.eval_tester.run(self.tokenizer, self.actor)
            print(f"Validation pass rate: {val_rate:.2f}%")
        except Exception as e:
            print(f"Validation evaluation failed: {e}")

        print("Running full evaluation on test split...")
        try:
            test_rate = self.test_tester.run(self.tokenizer, self.actor)
            print(f"Test pass rate: {test_rate:.2f}%")
        except Exception as e:
            print(f"Test evaluation failed: {e}")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cur_time = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    exec_log_file = f"execution_log_{cur_time}.csv"
    os.environ["EXECUTION_LOG_FILE"] = exec_log_file
    os.remove(exec_log_file) if os.path.exists(exec_log_file) else None
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(True)  # We need gradients for training
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    cfg = PPOConfig()
    print(f"PPO Configuration:")
    print(f"  - Learning rate schedule: {cfg.lr_schedule} ({cfg.init_lr} → {cfg.final_lr})")
    print(f"  - GAE normalization: {cfg.use_gae_normalization}")
    print(f"  - Value loss clipping: {cfg.value_loss_clipping}")
    print(f"  - Updates: {cfg.updates}, Epochs per update: {cfg.epochs}")
    print(f"  - Num envs: {cfg.num_envs}, Batch size: {cfg.batch_size}")
    
    trainer = PPOTrainer(cfg, device, log_file=exec_log_file)
    trainer.train(updates=cfg.updates)