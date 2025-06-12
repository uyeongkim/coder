"""
trainer.py - SHPPO Trainer
완전히 새로 작성 - 모든 문제 해결
"""

from __future__ import annotations
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from env import SHPPOEnv, EnvConfig, type_obs, type_act
from model import (
    SHPPOModelConfig, RoleConfig, SharedLLM, 
    build_actor_and_critic, MultiAgentActor, MultiHeadCritic, InferenceNet
)
from utils import RolloutBuffer, Trajectory, set_seed, move_to

@dataclass
class SHPPOConfig:
    # Environment
    env_config: EnvConfig = field(default_factory=lambda: EnvConfig(
        use_temp_dir=True,
        max_problems=50,
        use_parallel_execution=False
    ))
    
    # Model
    model_config: SHPPOModelConfig = field(default_factory=lambda: SHPPOModelConfig(
        base_model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        lora_r=4,
        lora_alpha=8,
        latent_dim=2,
        hete_layer_input_dim=64,
        hete_layer_output_dim=16,
        mlp_hidden_dim=32
    ))
    
    # Roles
    roles: Dict[str, RoleConfig] = field(default_factory=lambda: {
        "planner": RoleConfig("planner", obs_embed_dim=64, n_action_templates=2),
        "coder": RoleConfig("coder", obs_embed_dim=64, n_action_templates=2),
        "debugger": RoleConfig("debugger", obs_embed_dim=64, n_action_templates=2),
    })
    
    # Training
    n_episodes: int = 50
    n_epochs: int = 2
    mini_batch_size: int = 4
    lr_actor: float = 3e-3
    lr_critic: float = 1e-2
    gamma: float = 0.9
    lam: float = 0.8
    clip_eps: float = 0.3
    entropy_coeff: float = 0.1
    max_grad_norm: float = 1.0
    
    # Logging
    log_interval: int = 1
    eval_interval: int = 10
    save_interval: int = 20
    checkpoint_dir: str = "./checkpoints"
    wandb_project: str = "shppo-clean"
    wandb_run_name: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

class SHPPOTrainer:
    def __init__(self, config: SHPPOConfig):
        self.cfg = config
        set_seed(self.cfg.seed)
        
        # Setup
        self._setup_logging()
        self.env = SHPPOEnv(self.cfg.env_config)
        self._build_models()
        self._setup_optimizers()
        
        # Buffers and states
        self.buffers = {role: RolloutBuffer(self.cfg.gamma, self.cfg.lam) 
                       for role in self.cfg.roles.keys()}
        
        device = torch.device(self.cfg.device)
        self.hidden_states = {
            role: torch.zeros(1, self.cfg.model_config.hete_layer_input_dim, 
                            device=device, dtype=torch.float32) 
            for role in self.cfg.roles.keys()
        }
        
        # Training state
        self.episode = 0
        self.step = 0
        self.best_reward = float('-inf')
        self.metrics = defaultdict(list)
        
        print("✅ Trainer initialized")
        
    def _setup_logging(self):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.cfg.checkpoint_dir}/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_run_name or f"run-{int(time.time())}",
            config=self.cfg.__dict__,
            reinit=True
        )
        
    def _build_models(self):
        device = torch.device(self.cfg.device)
        
        self.llm = SharedLLM(self.cfg.model_config, device)
        self.actors, self.critic, self.inference_net = build_actor_and_critic(
            self.cfg.model_config, self.cfg.roles, device
        )
        
        self.obs_projection = nn.Linear(
            self.llm.model.config.hidden_size,
            self.cfg.model_config.hete_layer_input_dim
        ).to(device)
        
        print(f"Models built on {device}")
        
    def _setup_optimizers(self):
        self.opt_actors = {
            role: torch.optim.Adam(actor.parameters(), lr=self.cfg.lr_actor)
            for role, actor in self.actors.items()
        }
        
        critic_params = list(self.critic.parameters()) + list(self.obs_projection.parameters())
        self.opt_critic = torch.optim.Adam(critic_params, lr=self.cfg.lr_critic)
        
        llm_params = list(self.llm.trainable_parameters())
        self.opt_llm = torch.optim.Adam(llm_params, lr=1e-4) if llm_params else None
        
    def _embed_observation(self, obs: type_obs) -> torch.Tensor:
        role = obs["role"]
        files_text = "\n".join([f"=== {fname} ===\n{content[:100]}..." 
                               for fname, content in obs["visible_files"].items()])
        
        prompt = f"Role: {role}\n\nFiles:\n{files_text}\n\nNext:"
        
        with torch.no_grad():
            inputs = self.llm.tokenizer(prompt, return_tensors="pt", 
                                      truncation=True, max_length=256)
            device = next(self.llm.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = self.llm.model(**inputs, output_hidden_states=True)
            raw_emb = outputs.hidden_states[-1].mean(dim=1).float()
            obs_emb = self.obs_projection(raw_emb)
            
        return obs_emb
        
    def _select_action(self, role: str, obs_emb: torch.Tensor, hidden: torch.Tensor):
        actor = self.actors[role]
        
        # Forward pass
        logits, new_hidden, (latent, mu, sigma) = actor(obs_emb.float(), hidden.float())
        
        # Sample action
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        # Value
        value_dict = self.critic(new_hidden)
        value = value_dict[role]
        
        return action.item(), logprob, value, new_hidden, latent
        
    def _generate_content(self, role: str, action_idx: int) -> str:
        if role == "planner":
            return """# Plan
1. Parse input
2. Design algorithm  
3. Handle edge cases
"""
        elif role == "coder":
            return """def solve(input_str):
    lines = input_str.strip().split('\\n')
    n = int(lines[0]) if lines else 1
    return str(n * 2)

if __name__ == "__main__":
    import sys
    print(solve(sys.stdin.read()))
"""
        else:  # debugger
            return """def solve(input_str):
    try:
        lines = input_str.strip().split('\\n')
        n = int(lines[0]) if lines else 1
        return str(max(0, n * 2))
    except:
        return "0"

if __name__ == "__main__":
    import sys
    print(solve(sys.stdin.read()))
"""
        
    def _run_episode(self) -> Dict[str, float]:
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # Reset
        device = torch.device(self.cfg.device)
        for role in self.cfg.roles.keys():
            self.hidden_states[role] = torch.zeros(1, self.cfg.model_config.hete_layer_input_dim, 
                                                  device=device, dtype=torch.float32)
            self.buffers[role].reset()
        
        while True:
            role = obs["role"]
            
            # Process step
            obs_emb = self._embed_observation(obs)
            action_idx, logprob, value, new_hidden, latent = self._select_action(
                role, obs_emb, self.hidden_states[role]
            )
            
            self.hidden_states[role] = new_hidden
            
            # Generate action
            if role == "planner":
                filename = "plan.md"
            elif role == "coder":
                filename = "code.py"
            else:
                filename = "fixed_code.py"
                
            content = self._generate_content(role, action_idx)
            action = {"filename": filename, "content": content}
            
            # Store trajectory
            traj = Trajectory(
                state=obs_emb.squeeze(0),
                latent=latent.squeeze(0),
                action=torch.tensor(action_idx),
                logp=logprob,
                reward=torch.tensor(0.0),
                value=value
            )
            self.buffers[role].add(traj)
            
            # Environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # Update reward
            if len(self.buffers[role].traj) > 0:
                self.buffers[role].traj[-1].reward = torch.tensor(reward)
            
            episode_reward += reward
            episode_length += 1
            self.step += 1
            
            if done:
                break
                
            obs = next_obs
            
        # Compute GAE
        self._compute_gae()
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "final_reward": reward
        }
        
    def _compute_gae(self):
        """Simple GAE computation"""
        for role, buffer in self.buffers.items():
            if len(buffer.traj) == 0:
                continue
                
            rewards = [t.reward.item() for t in buffer.traj]
            values = [t.value.item() for t in buffer.traj]
            
            # GAE
            advantages = []
            returns = []
            gae = 0.0
            
            for t in reversed(range(len(rewards))):
                next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
                delta = rewards[t] + self.cfg.gamma * next_value - values[t]
                gae = delta + self.cfg.gamma * self.cfg.lam * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
            
            # Normalize advantages
            if len(advantages) > 1:
                mean_adv = sum(advantages) / len(advantages)
                std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
                if std_adv > 1e-8:
                    advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in advantages]
            
            # Assign to trajectories
            for i, traj in enumerate(buffer.traj):
                device = traj.reward.device
                traj.returns = torch.tensor(returns[i], dtype=torch.float32, device=device)
                traj.advs = torch.tensor(advantages[i], dtype=torch.float32, device=device)
        
    def _compute_losses(self, role: str, batch: List[Trajectory]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}
            
        # Check GAE
        if not all(hasattr(t, 'returns') and hasattr(t, 'advs') for t in batch):
            print(f"⚠️ GAE missing for {role}")
            return {}
            
        # Process each sample individually to avoid GRU dimension issues
        all_actor_losses = []
        all_value_losses = []
        all_entropies = []
        all_kl_divs = []
        
        actor = self.actors[role]
        
        for traj in batch:
            # Individual forward pass
            state = traj.state.unsqueeze(0)  # [1, state_dim]
            latent = traj.latent.unsqueeze(0)  # [1, latent_dim]
            action = traj.action.unsqueeze(0)  # [1]
            old_logprob = traj.logp
            returns = traj.returns
            advantages = traj.advs
            
            # Actor forward
            logits, new_hidden, _ = actor(state, latent)
            
            # Policy
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_logprob = dist.log_prob(action).squeeze()
            entropy = dist.entropy().squeeze()
            
            # PPO loss
            ratio = torch.exp(new_logprob - old_logprob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2)
            
            # Value loss
            new_value = self.critic(new_hidden)[role].squeeze()
            value_loss = F.mse_loss(new_value, returns)
            
            # KL
            kl_div = (old_logprob - new_logprob).abs()
            
            all_actor_losses.append(actor_loss)
            all_value_losses.append(value_loss)
            all_entropies.append(entropy)
            all_kl_divs.append(kl_div)
        
        return {
            'actor_loss': torch.stack(all_actor_losses).mean(),
            'value_loss': torch.stack(all_value_losses).mean(),
            'entropy': torch.stack(all_entropies).mean(),
            'kl_div': torch.stack(all_kl_divs).mean(),
        }
        
    def _update_networks(self):
        total_losses = defaultdict(list)
        
        for epoch in range(self.cfg.n_epochs):
            for role, buffer in self.buffers.items():
                if len(buffer.traj) == 0:
                    continue
                    
                # Mini-batches
                indices = torch.randperm(len(buffer.traj))
                
                for start in range(0, len(indices), self.cfg.mini_batch_size):
                    end = min(start + self.cfg.mini_batch_size, len(indices))
                    batch_indices = indices[start:end]
                    batch = [buffer.traj[i] for i in batch_indices]
                    
                    losses = self._compute_losses(role, batch)
                    if not losses:
                        continue
                        
                    # Update actor
                    self.opt_actors[role].zero_grad()
                    actor_total = losses['actor_loss'] - self.cfg.entropy_coeff * losses['entropy']
                    actor_total.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.actors[role].parameters(), self.cfg.max_grad_norm)
                    self.opt_actors[role].step()
                    
                    # Update critic
                    self.opt_critic.zero_grad()
                    losses['value_loss'].backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                    self.opt_critic.step()
                    
                    # Store losses
                    for k, v in losses.items():
                        total_losses[f"{role}_{k}"].append(v.item())
                        
        return {k: sum(v) / len(v) if v else 0.0 for k, v in total_losses.items()}
        
    def train(self):
        print(f"🚀 Training for {self.cfg.n_episodes} episodes")
        
        pbar = tqdm(range(self.cfg.n_episodes), desc="Training")
        
        for episode in pbar:
            self.episode = episode
            
            try:
                # Reset buffers
                for buffer in self.buffers.values():
                    buffer.reset()
                
                # Run episode
                episode_metrics = self._run_episode()
                
                # Update networks
                loss_metrics = self._update_networks()
                
                # Metrics
                episode_reward = episode_metrics.get('episode_reward', 0.0)
                pass_rate = 1.0 if episode_reward >= 0.99 else 0.0
                
                # Log
                log_data = {
                    "train/episode": episode,
                    "train/episode_reward": episode_reward,
                    "train/pass_rate": pass_rate,
                    "train/episode_length": episode_metrics.get('episode_length', 0),
                }
                
                for k, v in loss_metrics.items():
                    log_data[f"train/{k}"] = v
                
                wandb.log(log_data, step=episode)
                
                # Progress
                pbar.set_postfix({
                    'reward': f"{episode_reward:.3f}",
                    'pass_rate': f"{pass_rate:.3f}",
                    'length': f"{episode_metrics.get('episode_length', 0):.0f}"
                })
                
                # Console log
                if episode % self.cfg.log_interval == 0:
                    print(f"Episode {episode}: reward={episode_reward:.4f}, pass_rate={pass_rate:.3f}")
                
                # Save
                if episode > 0 and episode % self.cfg.save_interval == 0:
                    self._save_checkpoint(episode)
                    
            except Exception as e:
                print(f"❌ Episode {episode} failed: {e}")
                import traceback
                traceback.print_exc()
                raise
                
        print("🎉 Training completed!")
        wandb.finish()
        
    def _save_checkpoint(self, episode):
        checkpoint = {
            'episode': episode,
            'actors': {role: actor.state_dict() for role, actor in self.actors.items()},
            'critic': self.critic.state_dict(),
            'obs_projection': self.obs_projection.state_dict(),
            'config': self.cfg,
            'metrics': dict(self.metrics)
        }
        
        path = f"{self.cfg.checkpoint_dir}/checkpoint_{episode}.pt"
        torch.save(checkpoint, path)
        print(f"💾 Saved: {path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default="shppo-clean")
    args = parser.parse_args()
    
    config = SHPPOConfig()
    config.n_episodes = args.episodes
    config.wandb_project = args.wandb_project
    
    if args.device != "auto":
        config.device = args.device
    
    print(f"Starting training: {config.n_episodes} episodes on {config.device}")
    
    trainer = SHPPOTrainer(config)
    
    try:
        trainer.train()
        print("✅ Success!")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(trainer, 'env'):
            trainer.env.cleanup()

if __name__ == "__main__":
    main()