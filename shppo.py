#!/usr/bin/env python3
"""
A full PPO training script using Qwen-LoRA as actor and a custom Critic,
with CodeTester (from test.py) as the environment. Single‐step episodes,
vectorized across multiple CodeTester instances.
"""

import os
import math
import torch
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
    updates: int = 1000  # Number of PPO updates

    # PPO update
    epochs: int = 1
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 2         # minibatch size for PPO (<= num_envs * rollout_length)

    # Optimizer
    learning_rate: float = 3e-4

    # Training
    max_problems_per_update: int = 10  # Number of problems to sample per update
    
    # Model Configuration
    lora_r = 8
    lora_alpha = 16

    def __post_init__(self):
        assert self.num_envs * self.rollout_length >= self.batch_size, \
            "Batch size must be less than or equal to num_envs * rollout_length, " \
            f"got {self.batch_size} > {self.num_envs * self.rollout_length}"

def ortho(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
# ppo.py - SHPPO 구조 기반 통합 Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import itertools
from typing import Optional, Tuple

# ────────────────────────────────────── util init ─────────────────────────────────────
# ppo.py - SHPPO 구조 기반 통합 Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import itertools
from typing import Optional, Tuple

# ppo.py - SHPPO 구조 기반 통합 Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from typing import Optional, Tuple

# ────────────────────────────────────── Utils ─────────────────────────────────────
def ortho_init(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, layers: int = 3):
        super().__init__()
        layers_list = []
        d = input_dim
        for _ in range(layers - 1):
            layers_list.append(nn.Linear(d, hidden_dim))
            layers_list.append(nn.ReLU())
            d = hidden_dim
        layers_list.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers_list)
        self.net.apply(lambda m: ortho_init(m, math.sqrt(2)))

    def forward(self, x):
        return self.net(x)

# ────────────────────────────────────── LatentNet ─────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, obs_embed_dim, rnn_hidden_dim, latent_dim, mlp_hidden_dim):
        super().__init__()
        input_dim = obs_embed_dim + rnn_hidden_dim
        self.encoder = MLPBlock(input_dim, mlp_hidden_dim, hidden_dim=mlp_hidden_dim, layers=2)
        self.fc_mu = nn.Linear(mlp_hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(mlp_hidden_dim, latent_dim)

    def forward(self, obs_embedding, h_actor_prev):
        x = torch.cat([obs_embedding, h_actor_prev], dim=-1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x)) + 1e-5
        return mu, sigma

class LatentNet(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs_embedding: torch.Tensor, h_actor_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.encoder(obs_embedding, h_actor_prev)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z, mu, sigma

# ────────────────────────────────────── InferenceNet ─────────────────────────────────────
class InferenceNet(nn.Module):
    def __init__(self, global_obs_dim, latent_dim, num_agents, mlp_hidden_dim):
        super().__init__()
        input_dim = global_obs_dim + 2 * num_agents * latent_dim
        self.v_head = MLPBlock(input_dim, 1, hidden_dim=mlp_hidden_dim, layers=3)

    def forward(self, global_obs, mus_all, sigmas_all):
        batch_size = global_obs.shape[0]
        mus_flat = mus_all.view(batch_size, -1)
        sigmas_flat = sigmas_all.view(batch_size, -1)
        x = torch.cat([global_obs, mus_flat, sigmas_flat], dim=-1)
        return self.v_head(x).squeeze(-1)

# ────────────────────────────────────── HeteLayerDecoder ─────────────────────────────────────
class HeteLayerDecoder(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int, output_dim: int):
        super().__init__()
        self.w_decoder = nn.Linear(latent_dim, input_dim * output_dim)
        self.b_decoder = nn.Linear(latent_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w_decoder.apply(lambda m: ortho_init(m, math.sqrt(0.1)))
        self.b_decoder.apply(lambda m: ortho_init(m, math.sqrt(0.1)))

    def forward(self, z):
        B = z.size(0)
        W = self.w_decoder(z).view(B, self.output_dim, self.input_dim)
        b = self.b_decoder(z)
        return W, b

# ────────────────────────────────────── ActorNet ─────────────────────────────────────
class ActorNet(nn.Module):
    def __init__(self, obs_dim, rnn_hidden_dim, latent_dim, hete_out_dim, final_out_dim, action_dim):
        super().__init__()
        self.obs_encoder = MLPBlock(obs_dim, rnn_hidden_dim, hidden_dim=128, layers=3)  # o_n → o_feat
        self.rnn = nn.GRU(rnn_hidden_dim, rnn_hidden_dim, batch_first=True)
        self.decoder = HeteLayerDecoder(latent_dim, rnn_hidden_dim, hete_out_dim)
        self.final_mlp = MLPBlock(
            input_dim=hete_out_dim,
            output_dim=final_out_dim,
            hidden_dim=final_out_dim,
            layers=3
        )
        self.policy_head = nn.Linear(final_out_dim, action_dim)

    def forward(self, obs, h_actor_prev, z):
        o_feat = self.obs_encoder(obs)
        rnn_out, h_actor_next = self.rnn(o_feat.unsqueeze(1), h_actor_prev)
        h_actor = rnn_out.squeeze(1)
        W, b = self.decoder(z)
        hete = torch.bmm(W, h_actor.unsqueeze(-1)).squeeze(-1) + b
        x = self.final_mlp(hete)
        logits = self.policy_head(x)
        return logits, h_actor_next

# ────────────────────────────────────── CriticNet ─────────────────────────────────────
class CriticNet(nn.Module):
    def __init__(self, global_dim: int, rnn_hidden_dim: int, n_agents: int):
        super().__init__()
        self.obs_proj = MLPBlock(global_dim, rnn_hidden_dim, hidden_dim=64, layers=2)
        self.rnn = nn.GRU(rnn_hidden_dim, rnn_hidden_dim, batch_first=True)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(rnn_hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(n_agents)
        ])
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_agents = n_agents

    def forward(self, global_obs: torch.Tensor, h_critic_prev: torch.Tensor, agent_ids: torch.Tensor):
        x = self.obs_proj(global_obs)
        rnn_out, h_critic_next = self.rnn(x.unsqueeze(1), h_critic_prev)
        h = rnn_out.squeeze(1)

        out = torch.zeros(h.size(0), device=h.device)
        for i in range(self.n_agents):
            mask = (agent_ids == i)
            if mask.any():
                out[mask] = self.heads[i](h[mask]).squeeze(-1)

        return out, h_critic_next



# ────────────────────────────────────── utils ─────────────────────────────────────
def cosine_diversity(z):
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
    mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    return (1 - sim[mask]).mean()

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
        # The run method returns (avg_rate, hidden_states, generated_tokens, log_probs)
        try:
            result = temp_tester.run(
                self.tokenizer,
                self.actor,
                return_hidden=True,
                return_logprobs=True
            )
            
            # Unpack the results
            if isinstance(result, tuple) and len(result) == 4:
                avg_rate, hidden_states, generated_tokens, log_probs = result
            else:
                raise ValueError(f"Expected 4 return values, got {len(result) if isinstance(result, tuple) else 1}")
            
            # Get the actual solutions from the responses
            # We need to extract the solutions from the generated tokens
            solutions = []
            input_lengths = []
            
            # Decode the generated tokens to get solutions
            for i in range(generated_tokens.shape[0]):
                token_ids = generated_tokens[i]
                # Remove padding tokens
                non_pad_tokens = token_ids[token_ids != self.tokenizer.pad_token_id]
                solution = self.tokenizer.decode(non_pad_tokens, skip_special_tokens=True)
                solutions.append(solution)
            
        except Exception as e:
            print(f"Error in temp_tester.run: {e}")
            # Fallback: generate solutions without using the temporary tester
            prompts = [problem['prompt'] for problem in problems]
            
            # Use the original tester's generation method
            if hasattr(self.code_tester, 'ask_qwen_batch_optimized'):
                result = self.code_tester.ask_qwen_batch_optimized(
                    self.tokenizer, 
                    self.actor, 
                    prompts, 
                    return_hidden=True, 
                    return_logprobs=True
                )
                solutions, hidden_states, generated_tokens, log_probs = result
            else:
                raise Exception("Unable to generate solutions with hidden states and log probs")
        
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
        """Compute advantages using GAE"""
        # For single-step episodes, advantage is simply reward - value
        advantages = rewards - values
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
            
            # For proper PPO, we need to compute current log probabilities
            # This is a simplified version - in practice, you'd need to recompute
            # the log probabilities by running the current policy on the same inputs
            
            # Simplified policy loss using advantage as proxy
            # In a full implementation, you would:
            # 1. Recompute log probabilities with current policy
            # 2. Compute ratio = exp(new_log_probs - old_log_probs)
            # 3. Apply PPO clipping to the ratio
            
            # For now, using advantage-weighted policy loss
            policy_loss = -advantages.mean()
            
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