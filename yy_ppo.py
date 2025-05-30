#!/usr/bin/env python3
"""
A full PPO training script using Qwen-LoRA as actor and a custom Critic,
with CodeTester (from test.py) as the environment. Single‐step episodes,
vectorized across multiple CodeTester instances.
"""

import os
import math
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim
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
    num_envs: int = 2
    rollout_length: int = 1      # single‐step episodes
    gamma: float = 0.99
    lam: float = 0.95
    updates: int = 10  # Number of PPO updates

    # PPO update
    epochs: int = 10
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 2         # minibatch size for PPO (<= num_envs * rollout_length)

    # Optimizer
    learning_rate: float = 3e-4

    # Training
    max_problems_per_update: int = 10  # Number of problems to sample per update
    sample_space: int = 4 # Action space size (number of test cases per problem)
    
    # Model Configuration
    lora_r: int = 8
    lora_alpha: int = 16

    def __post_init__(self):
        assert self.num_envs * self.rollout_length >= self.batch_size, \
            "Batch size must be less than or equal to num_envs * rollout_length, " \
            f"got {self.batch_size} > {self.num_envs * self.rollout_length}"

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
    def __init__(self, cfg: PPOConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        # Build actor, critic, tokenizer
        self.actor, self.critic, a_vars, c_vars, self.tokenizer = build_actor_and_critic(cfg, device)
        self.optimizer = optim.AdamW(
            a_vars + c_vars,
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        # Dataset and CodeTester
        self.dataset = CodeContestDataset(split="train", max_problems=100, max_cases=3)
        self.code_tester = CodeTester(
            dataset=self.dataset, 
            batch_size=cfg.num_envs, 
            max_workers=4
        )
        
        print(f"Loaded {len(self.dataset.get_all_tasks())} tasks for training")

    def sample_problems(self, num_problems: int) -> List[Dict[str, Any]]:
        """Sample random problems from the dataset"""
        all_tasks = self.dataset.get_all_tasks()
        return random.sample(all_tasks, min(num_problems, len(all_tasks)))

    def evaluate_solutions_with_tester(self) -> List[float]:
        """Evaluate solutions using the test cases, aggregate rewards per problem, and return per-problem rewards"""
        csv_file = "execution_results.csv"
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
    
    def extract_solve_function(self, response: str) -> callable:
        """Extract solve function from model response"""
        import re
        
        code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
        code_match = code_pattern.search(response)
        code_str = code_match.group(1).strip() if code_match else response.strip()
        
        # Quick validation before exec
        if not code_str or 'def solve' not in code_str:
            return None
            
        namespace = {}
        try:
            exec(code_str, namespace)
            return namespace.get("solve", None)
        except Exception:
            return None

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
            batch_size=self.cfg.num_envs,
            max_workers=4
        )

        # Generate solutions with hidden states and log probabilities
        # Use ask_qwen_batch_optimized instead of run()
        result = temp_tester.ask_qwen_batch_optimized(
            tokenizer=self.tokenizer,
            model=self.actor,
            prompts=[problem['prompt'] for problem in problems],
            return_hidden=True,
            return_logprobs=True,
            num_return_sequences=self.cfg.sample_space,
        )
        solutions, hidden_states, generated_tokens, log_probs = result

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
        log_probs = dist.log_prob(chosen_indices).to(self.device)  # shape: [batch_size]

        # Evaluate solutions to get rewards
        rewards = self.code_tester.evaluate_solutions_with_tester()

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
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * gae
            advantages[t] = gae
        return advantages

    def ppo_update(self, batch: Dict[str, torch.Tensor]):
        """Perform PPO update with proper policy loss computation"""
        rewards = batch["rewards"]
        values = batch["values"]
        hidden_states = batch["hidden_states"]
        old_log_probs = batch["log_probs"]
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, values)
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Value targets
        returns = rewards  # For single-step episodes
        
        # Multiple PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        
        for epoch in range(self.cfg.epochs):
            # Get current value estimates
            current_values = self.critic(hidden_states)
            
            # Value loss
            value_loss = nn.MSELoss()(current_values, returns)
            
            # PPO policy loss with clipping
            # Placeholder for recomputing new log_probs
            # (for now, assume we stored them in old_log_probs and don't change actor outputs)
            ratios = torch.exp(old_log_probs - old_log_probs.detach())
            clipped_ratios = torch.clamp(ratios, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
            
            # Simple entropy bonus (using log_probs as proxy)
            entropy_loss = -old_log_probs.mean() * self.cfg.ent_coef
            
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
            name=f"PPO-Training-{updates}Updates",
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
        print("Starting PPO training...")
        
        best_avg_reward = 0.0
        
        for update in range(1, updates + 1):
            # Collect rollouts
            batch = self.collect_rollouts()
            generation_table.add_data(
                update,
                batch["prompts"],
                batch["solutions"],
                batch["rewards"].cpu().numpy()
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
                      f"avg_reward={avg_reward:.4f}, "
                      f"max_reward={max_reward:.4f}, "
                      f"min_reward={min_reward:.4f}, "
                      f"policy_loss={policy_loss:.4f}, "
                      f"value_loss={value_loss:.4f}, "
                      f"entropy_loss={entropy_loss:.4f}, "
                      f"avg_log_prob={avg_log_prob:.4f}")
            wandb.log({
                "update": update,
                "avg_reward": avg_reward,
                "max_reward": max_reward,
                "min_reward": min_reward,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "avg_log_prob": avg_log_prob
            }, step=update)
            
            # Periodic full evaluation
            if update % 25 == 0:
                print(f"\n--- Full Evaluation at Update {update} ---")
                self.full_evaluation()
                print("--- End Evaluation ---\n")
            
            wandb.log({"generation": generation_table})
        print(f"Training complete. Best average reward: {best_avg_reward:.4f}")

    def full_evaluation(self):
        """Run a full evaluation using CodeTester.run()"""
        print("Running full evaluation...")
        
        # Use the original CodeTester.run method for comprehensive evaluation
        try:
            avg_pass_rate = self.code_tester.run(self.tokenizer, self.actor)
            print(f"Full evaluation pass rate: {avg_pass_rate:.2f}%")
        except Exception as e:
            print(f"Full evaluation failed: {e}")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.remove("execution_results.csv") if os.path.exists("execution_results.csv") else None
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(True)  # We need gradients for training
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    cfg = PPOConfig()
    trainer = PPOTrainer(cfg, device)
    trainer.train(updates=cfg.updates)