"""
trainer.py - SHPPO (Shared Heterogeneous PPO) Trainer
Multi-agent RL training with shared LLM backbone and role-specific policies
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

# ═══════════════════════════════════════════════════════════════════════════
# 1. Training Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SHPPOConfig:
    # Environment
    env_config: EnvConfig = field(default_factory=lambda: EnvConfig(root_dir="./tmp/workspace"))
    
    # Model architecture
    model_config: SHPPOModelConfig = field(default_factory=lambda: SHPPOModelConfig(
        base_model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",  # 더 작은 모델 사용
        lora_r=8,
        lora_alpha=16,
        latent_dim=4,  # 더 작은 잠재 차원
        hete_layer_input_dim=256,  # 더 작은 히든 차원
        hete_layer_output_dim=32,
        mlp_hidden_dim=128
    ))
    
    # Role definitions (더 작은 템플릿 수)
    roles: Dict[str, RoleConfig] = field(default_factory=lambda: {
        "planner": RoleConfig("planner", obs_embed_dim=256, n_action_templates=3),
        "coder": RoleConfig("coder", obs_embed_dim=256, n_action_templates=4),
        "debugger": RoleConfig("debugger", obs_embed_dim=256, n_action_templates=5),
    })
    
    # Training hyperparameters
    n_episodes: int = 1000
    n_epochs: int = 4
    mini_batch_size: int = 32
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_llm: float = 1e-5
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    kl_target: float = 0.01
    kl_coeff: float = 0.1
    max_grad_norm: float = 0.5
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    checkpoint_dir: str = "./checkpoints"
    wandb_project: str = "shppo-code-contests"
    wandb_run_name: Optional[str] = None
    
    # Device and reproducibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Action templates for each role
    action_templates: Dict[str, List[str]] = field(default_factory=lambda: {
        "planner": [
            "Analyze the problem and break it down into steps: {analysis}",
            "Identify key constraints and edge cases: {constraints}",
            "Propose algorithmic approach: {algorithm}",
            "Suggest implementation strategy: {strategy}"
        ],
        "coder": [
            "Implement the solution: {code}",
            "Add input parsing logic: {parsing}",
            "Implement core algorithm: {algorithm}",
            "Add error handling: {error_handling}",
            "Optimize for performance: {optimization}",
            "Add debugging prints: {debug}"
        ],
        "debugger": [
            "Analyze test failures: {analysis}",
            "Fix logical errors: {logic_fix}",
            "Correct edge case handling: {edge_fix}",
            "Optimize algorithm: {optimization}",
            "Fix parsing issues: {parsing_fix}",
            "Add missing imports: {imports}",
            "Improve error handling: {error_handling}",
            "Final code review: {review}"
        ]
    })

# ═══════════════════════════════════════════════════════════════════════════
# 2. SHPPO Trainer Class
# ═══════════════════════════════════════════════════════════════════════════

class SHPPOTrainer:
    def __init__(self, config: SHPPOConfig):
        self.cfg = config
        set_seed(self.cfg.seed)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize environment
        self.env = SHPPOEnv(self.cfg.env_config)
        
        # Initialize models
        self._build_models()
        
        # Initialize optimizers
        self._setup_optimizers()
        
        # Initialize buffers and tracking
        self.buffers = {role: RolloutBuffer(self.cfg.gamma, self.cfg.lam) 
                       for role in self.cfg.roles.keys()}
        
        # Initialize hidden states on the correct device with correct dtype
        device = torch.device(self.cfg.device)
        self.hidden_states = {role: torch.zeros(1, self.cfg.model_config.hete_layer_input_dim, 
                                               device=device, dtype=torch.float32) 
                             for role in self.cfg.roles.keys()}
        
        # Training state
        self.episode = 0
        self.step = 0
        self.best_reward = float('-inf')
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        
    def _setup_logging(self):
        """Initialize logging and wandb."""
        import wandb
        
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
        
        # Initialize wandb
        wandb_config = {
            "model_config": self.cfg.model_config.__dict__,
            "env_config": self.cfg.env_config.__dict__,
            "training": {
                "n_episodes": self.cfg.n_episodes,
                "n_epochs": self.cfg.n_epochs,
                "mini_batch_size": self.cfg.mini_batch_size,
                "lr_actor": self.cfg.lr_actor,
                "lr_critic": self.cfg.lr_critic,
                "lr_llm": self.cfg.lr_llm,
                "gamma": self.cfg.gamma,
                "lam": self.cfg.lam,
                "clip_eps": self.cfg.clip_eps,
                "entropy_coeff": self.cfg.entropy_coeff,
                "value_loss_coeff": self.cfg.value_loss_coeff,
            },
            "roles": {name: rcfg.__dict__ for name, rcfg in self.cfg.roles.items()},
            "device": self.cfg.device,
            "seed": self.cfg.seed,
        }
        
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_run_name or f"shppo-{self.cfg.seed}",
            config=wandb_config,
            reinit=True
        )
        
        self.logger.info("Wandb initialized successfully")
        
    def _build_models(self):
        """Initialize all models and move to device."""
        device = torch.device(self.cfg.device)
        
        # Shared LLM
        self.llm = SharedLLM(self.cfg.model_config, device)
        
        # Actor-critic networks - pass roles to build function
        self.actors, self.critic, self.inference_net = build_actor_and_critic(
            self.cfg.model_config, self.cfg.roles, device
        )
        
        self.logger.info(f"Models initialized on {device}")
        self._log_model_parameters()
        
    def _log_model_parameters(self):
        """Log parameter counts for each component."""
        llm_params = sum(p.numel() for p in self.llm.trainable_parameters())
        actor_params = sum(sum(p.numel() for p in actor.parameters()) 
                          for actor in self.actors.values())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        inf_params = sum(p.numel() for p in self.inference_net.parameters())
        
        self.logger.info(f"Parameter counts:")
        self.logger.info(f"  LLM (trainable): {llm_params:,}")
        self.logger.info(f"  Actors: {actor_params:,}")
        self.logger.info(f"  Critic: {critic_params:,}")
        self.logger.info(f"  Inference Net: {inf_params:,}")
        
    def _setup_optimizers(self):
        """Initialize optimizers for different components."""
        self.opt_actors = {
            role: torch.optim.Adam(actor.parameters(), lr=self.cfg.lr_actor)
            for role, actor in self.actors.items()
        }
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)
        self.opt_inference = torch.optim.Adam(self.inference_net.parameters(), lr=self.cfg.lr_critic)
        
        # LLM optimizer - only create if there are trainable parameters
        llm_params = list(self.llm.trainable_parameters())
        if llm_params:
            self.opt_llm = torch.optim.Adam(llm_params, lr=self.cfg.lr_llm)
            self.logger.info(f"LLM optimizer created with {len(llm_params)} parameters")
        else:
            self.opt_llm = None
            self.logger.warning("No trainable LLM parameters found, skipping LLM optimizer")
            
        # Debug: print parameter details
        for name, param_group in [
            ("LLM", llm_params),
            ("Actor (planner)", list(self.actors["planner"].parameters())),
            ("Critic", list(self.critic.parameters())),
            ("Inference", list(self.inference_net.parameters()))
        ]:
            trainable = [p for p in param_group if p.requires_grad]
            self.logger.info(f"{name}: {len(trainable)}/{len(param_group)} trainable parameters")
        
    def _embed_observation(self, obs: type_obs) -> torch.Tensor:
        """Convert observation to embedding using the shared LLM."""
        # Create prompt from observation
        role = obs["role"]
        files_text = "\n".join([f"=== {fname} ===\n{content}" 
                               for fname, content in obs["visible_files"].items()])
        
        prompt = f"Role: {role}\n\nAvailable files:\n{files_text}\n\nWhat should I do next?"
        
        # Get embedding from LLM (use hidden states)
        with torch.no_grad():
            inputs = self.llm.tokenizer(prompt, return_tensors="pt", 
                                      truncation=True, max_length=512)
            # Move inputs to the same device as the model
            device = next(self.llm.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get last hidden state as observation embedding
            outputs = self.llm.model(**inputs, output_hidden_states=True)
            # Use mean pooling of last hidden states and convert to float32
            obs_emb = outputs.hidden_states[-1].mean(dim=1).float()  # [1, hidden_dim]
            
        return obs_emb
        
    def _select_action(self, role: str, obs_emb: torch.Tensor, hidden: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Select action using role-specific actor with latent learning."""
        actor = self.actors[role]
        
        # Ensure consistent dtypes (float32)
        obs_emb = obs_emb.float()
        hidden = hidden.float()
        
        # Forward pass through actor to get latent and action logits
        logits, new_hidden, (latent, mu, sigma) = actor(obs_emb, hidden)
        
        # Sample action (template index)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        # Compute value estimate using global hidden state
        value_dict = self.critic(new_hidden)
        value = value_dict[role]
        
        # Prepare extras with all necessary information
        extras = {
            'latent': latent,
            'mu': mu, 
            'sigma': sigma,
            'probs': probs,
            'new_hidden': new_hidden
        }
        
        return action.item(), logprob, value, extras
        
    def _generate_action_content(self, role: str, template_idx: int, obs: type_obs) -> str:
        """Generate actual action content using LLM and selected template."""
        template = self.cfg.action_templates[role][template_idx]
        
        # Create generation prompt
        files_text = "\n".join([f"=== {fname} ===\n{content}" 
                               for fname, content in obs["visible_files"].items()])
        
        prompt = f"""Role: {role}

Available files:
{files_text}

Task: {template}

Generate the appropriate content for this action:"""
        
        # Generate using LLM with proper device handling
        try:
            response = self.llm.generate([prompt])
            content = response['texts'][0].strip()
        except Exception as e:
            self.logger.warning(f"LLM generation failed: {e}, using fallback")
            # Simple fallback content
            if role == "planner":
                content = f"Plan: {template}\n\nSteps:\n1. Analyze problem\n2. Design solution\n3. Implementation strategy"
            elif role == "coder":
                content = f"def solve(input_str):\n    # {template}\n    pass"
            else:  # debugger
                content = f"# Debug: {template}\n# TODO: Fix the code based on test results"
        
        return content
        
    def _run_episode(self) -> Dict[str, float]:
        """Run a single episode and collect trajectories."""
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # Reset hidden states and buffers
        device = torch.device(self.cfg.device)
        for role in self.cfg.roles.keys():
            self.hidden_states[role] = torch.zeros(1, self.cfg.model_config.hete_layer_input_dim, 
                                                  device=device, dtype=torch.float32)
            self.buffers[role].reset()
            
        while True:
            role = obs["role"]
            
            # Embed observation
            obs_emb = self._embed_observation(obs)
            
            # Select action template and get latent representation
            action_idx, logprob, value, extras = self._select_action(
                role, obs_emb, self.hidden_states[role]
            )
            
            # Update hidden state
            self.hidden_states[role] = extras.get('new_hidden', self.hidden_states[role])
            
            # Generate action content using LLM
            if role == "planner":
                filename = "plan.md"
            elif role == "coder":
                filename = "code.py"
            else:  # debugger
                filename = "fixed_code.py"
                
            content = self._generate_action_content(role, action_idx, obs)
            action = {"filename": filename, "content": content}
            
            # Store trajectory in role-specific buffer
            traj = Trajectory(
                state=obs_emb.squeeze(0),
                latent=extras['latent'].squeeze(0),  # Store the actual latent variable
                action=torch.tensor(action_idx),
                logp=logprob,
                reward=torch.tensor(0.0),  # Will be updated later
                value=value
            )
            self.buffers[role].add(traj)
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Update the last trajectory's reward
            if len(self.buffers[role].traj) > 0:
                self.buffers[role].traj[-1].reward = torch.tensor(reward)
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            self.step += 1
            
            if done:
                break
                
            obs = next_obs
            
        # Compute GAE for all roles that were used
        for role, buffer in self.buffers.items():
            if len(buffer.traj) > 0:
                buffer.compute_gae(last_value=0.0)
                
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "final_reward": reward
        }
        
    def _compute_losses(self, role: str, batch: List[Trajectory]) -> Dict[str, torch.Tensor]:
        """Compute PPO + SHPPO losses for a batch of trajectories."""
        if not batch:
            return {}
            
        # Stack batch data
        states = torch.stack([t.state for t in batch])
        latents = torch.stack([t.latent for t in batch])  
        actions = torch.stack([t.action for t in batch])
        old_logprobs = torch.stack([t.logp for t in batch])
        returns = torch.stack([t.returns for t in batch]) if hasattr(batch[0], 'returns') else None
        advantages = torch.stack([t.advs for t in batch]) if hasattr(batch[0], 'advs') else None
        old_values = torch.stack([t.value for t in batch])
        
        if returns is None or advantages is None:
            return {}
            
        # Forward pass through actor to get new policy and latent variables
        actor = self.actors[role]
        logits, new_hidden, (new_latent, mu, sigma) = actor(states, latents)
        
        # Compute new policy probabilities
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # PPO clipped objective
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        new_values = self.critic(new_hidden)[role]
        value_loss = F.mse_loss(new_values, returns)
        
        # KL divergence for adaptive clip
        kl_div = (old_logprobs - new_logprobs).mean()
        
        # === SHPPO Latent Learning Losses (from paper) ===
        
        # L_v: Inference network evaluation of latent variables
        # Combine all agent latents for global assessment
        global_obs = states.mean(dim=0, keepdim=True)  # Simplified global observation
        inference_value = self.inference_net(mu, sigma, new_hidden.mean(dim=0, keepdim=True))
        
        # L_e: Entropy regularization for identifiable latents
        latent_dists = torch.distributions.Normal(mu, sigma)
        entropy_loss = -latent_dists.entropy().mean()
        
        # L_d: Distance loss for diverse latents
        n_agents = mu.size(0)
        if n_agents > 1:
            # Compute pairwise cosine similarities
            mu_flat = mu.view(n_agents, -1)
            similarities = F.cosine_similarity(mu_flat.unsqueeze(1), mu_flat.unsqueeze(0), dim=2)
            # Remove diagonal (self-similarities)
            mask = ~torch.eye(n_agents, dtype=torch.bool, device=mu.device)
            similarities = similarities[mask]
            # Normalize distances (1 - similarity)
            distances = 1 - similarities
            if distances.numel() > 0:
                distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
                diversity_loss = -distances.mean()
            else:
                diversity_loss = torch.tensor(0.0, device=mu.device)
        else:
            diversity_loss = torch.tensor(0.0, device=mu.device)
        
        # Combined SHPPO latent loss
        latent_loss = (
            -inference_value.mean() +  # Maximize inference value (-L_v)
            self.cfg.entropy_coeff * entropy_loss +  # λ_e * L_e
            0.1 * diversity_loss  # λ_d * L_d (using 0.1 as default weight)
        )
        
        return {
            'actor_loss': actor_loss,
            'value_loss': value_loss,
            'entropy': entropy.mean(),
            'kl_div': kl_div.abs(),
            'latent_loss': latent_loss,
            'inference_value': inference_value.mean(),
            'entropy_reg': entropy_loss,
            'diversity_loss': diversity_loss,
            'ratio_mean': ratio.mean(),
            'ratio_std': ratio.std()
        }
        
    def _update_networks(self):
        """Update all networks using collected trajectories with SHPPO losses."""
        total_losses = defaultdict(list)
        
        # Collect all trajectories across roles for global latent learning
        all_trajectories = []
        for role, buffer in self.buffers.items():
            if len(buffer.traj) > 0:
                all_trajectories.extend([(role, traj) for traj in buffer.traj])
        
        if not all_trajectories:
            return {}
        
        for epoch in range(self.cfg.n_epochs):
            for role, buffer in self.buffers.items():
                if len(buffer.traj) == 0:
                    continue
                    
                # Create mini-batches
                indices = torch.randperm(len(buffer.traj))
                
                for start in range(0, len(indices), self.cfg.mini_batch_size):
                    end = min(start + self.cfg.mini_batch_size, len(indices))
                    batch_indices = indices[start:end]
                    batch = [buffer.traj[i] for i in batch_indices]
                    
                    # Compute losses
                    losses = self._compute_losses(role, batch)
                    if not losses:
                        continue
                        
                    # Update actor with PPO + latent losses
                    self.opt_actors[role].zero_grad()
                    actor_total_loss = (
                        losses['actor_loss'] + 
                        self.cfg.entropy_coeff * (-losses['entropy']) +
                        0.1 * losses['latent_loss']  # Add latent learning
                    )
                    actor_total_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.actors[role].parameters(), self.cfg.max_grad_norm)
                    self.opt_actors[role].step()
                    
                    # Update critic
                    self.opt_critic.zero_grad()
                    losses['value_loss'].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                    self.opt_critic.step()
                    
                    # Update inference network (global latent evaluation)
                    if 'inference_value' in losses:
                        self.opt_inference.zero_grad()
                        # Inference network learns to predict returns from latent variables
                        inference_loss = F.mse_loss(
                            losses['inference_value'], 
                            torch.stack([t.returns for t in batch]).mean()
                        )
                        inference_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.inference_net.parameters(), self.cfg.max_grad_norm)
                        self.opt_inference.step()
                    
                    # Store losses
                    for k, v in losses.items():
                        if isinstance(v, torch.Tensor):
                            total_losses[f"{role}_{k}"].append(v.item())
                        else:
                            total_losses[f"{role}_{k}"].append(v)
                            
        # Update LLM less frequently (shared across all roles)
        if self.episode % 10 == 0 and self.opt_llm is not None:
            self.opt_llm.zero_grad()
            # You could add LLM-specific losses here based on generation quality
            self.opt_llm.step()
            
        return {k: sum(v) / len(v) if v else 0.0 for k, v in total_losses.items()}
        
    def train(self):
        """Main training loop with wandb logging style from paste.txt"""
        self.logger.info("Starting SHPPO training...")
        
        pbar = tqdm(range(self.cfg.n_episodes), desc="Training")
        
        for episode in pbar:
            self.episode = episode
            
            # Reset buffers
            for buffer in self.buffers.values():
                buffer.reset()
                
            # Run episode
            episode_metrics = self._run_episode()
            
            # Update networks
            loss_metrics = self._update_networks()
            
            # Combine metrics
            all_metrics = {**episode_metrics, **loss_metrics}
            
            # Update tracking
            for k, v in all_metrics.items():
                self.metrics[k].append(v)
            
            # Calculate additional metrics (following paste.txt style)
            episode_reward = episode_metrics.get('episode_reward', 0.0)
            pass_rate = 1.0 if episode_reward >= 0.99 else 0.0  # Simple pass criteria
            error_rate = 1.0 if episode_reward < 0 else 0.0
            positive_rate = 1.0 if episode_reward > 0 else 0.0
            
            # Wandb logging (matching paste.txt format)
            log_data = {
                "train/step": self.step,
                "train/episode": episode,
                "train/lr": self.opt_actors[list(self.opt_actors.keys())[0]].param_groups[0]['lr'],
                "train/average_reward": episode_reward,
                "train/pass_rate": pass_rate,
                "train/error_rate": error_rate,
                "train/positive_rate": positive_rate,
                "train/episode_length": episode_metrics.get('episode_length', 0),
                "train/env_type": "code_contests",
                "train/framework": "SHPPO",
            }
            
            # Add loss metrics
            for k, v in loss_metrics.items():
                if isinstance(v, (int, float)):
                    log_data[f"train/{k}"] = v
            
            # Add role-specific metrics
            for role in self.cfg.roles.keys():
                role_metrics = {k: v for k, v in loss_metrics.items() if k.startswith(f"{role}_")}
                for k, v in role_metrics.items():
                    log_data[f"train/{k}"] = v
            
            # Log to wandb
            wandb.log(log_data, step=episode)
                
            # Console logging
            if episode % self.cfg.log_interval == 0:
                msg = (f"Episode {episode}: "
                      f"reward={episode_reward:.3f}, "
                      f"pass_rate={pass_rate:.3f}, "
                      f"length={episode_metrics.get('episode_length', 0):.0f}")
                self.logger.info(msg)
                
            # Evaluation
            if episode % self.cfg.eval_interval == 0:
                eval_metrics = self._evaluate()
                eval_log_data = {}
                for k, v in eval_metrics.items():
                    eval_log_data[f"eval/{k}"] = v
                eval_log_data["eval/step"] = episode
                wandb.log(eval_log_data, step=episode)
                
            # Checkpointing
            if episode % self.cfg.save_interval == 0:
                self._save_checkpoint(episode)
                
            # Update progress bar
            pbar.set_postfix({
                'reward': f"{episode_reward:.3f}",
                'pass_rate': f"{pass_rate:.3f}",
                'length': f"{episode_metrics.get('episode_length', 0):.0f}"
            })
            
        self.logger.info("Training completed!")
        self._save_checkpoint(self.cfg.n_episodes)
        wandb.finish()
        
    def _evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate current policy."""
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(n_episodes):
            metrics = self._run_episode()
            eval_rewards.append(metrics['episode_reward'])
            eval_lengths.append(metrics['episode_length'])
            
        return {
            'mean_reward': sum(eval_rewards) / len(eval_rewards),
            'mean_length': sum(eval_lengths) / len(eval_lengths),
            'max_reward': max(eval_rewards),
            'min_reward': min(eval_rewards)
        }
        
    def _log_metrics(self, episode: int, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics to wandb."""
        # Convert metrics for wandb logging (following paste.txt style)
        wandb_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                wandb_metrics[f"{prefix}/{k}"] = v
        
        # Add step information
        wandb_metrics[f"{prefix}/step"] = episode
        wandb_metrics[f"{prefix}/episode"] = episode
        
        # Log to wandb
        wandb.log(wandb_metrics, step=episode)
            
        if prefix == "train" and episode % (self.cfg.log_interval * 5) == 0:
            # Log key metrics to console
            key_metrics = {k: v for k, v in metrics.items() if any(x in k.lower() for x in ['reward', 'loss', 'rate', 'pass'])}
            msg = f"Episode {episode}: " + ", ".join([f"{k}={v:.4f}" for k, v in key_metrics.items()][:5])
            self.logger.info(msg)
            
    def _save_checkpoint(self, episode: int):
        """Save model checkpoints."""
        checkpoint = {
            'episode': episode,
            'actors': {role: actor.state_dict() for role, actor in self.actors.items()},
            'critic': self.critic.state_dict(),
            'inference_net': self.inference_net.state_dict(),
            'llm': self.llm.model.state_dict(),
            'optimizers': {
                'actors': {role: opt.state_dict() for role, opt in self.opt_actors.items()},
                'critic': self.opt_critic.state_dict(),
                'inference': self.opt_inference.state_dict(),
                'llm': self.opt_llm.state_dict()
            },
            'config': self.cfg,
            'metrics': dict(self.metrics)
        }
        
        path = f"{self.cfg.checkpoint_dir}/checkpoint_{episode}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
        
        # Save best model
        current_reward = self.metrics.get('episode_reward', [0])[-1] if self.metrics.get('episode_reward') else 0
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            best_path = f"{self.cfg.checkpoint_dir}/best_model.pt"
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.cfg.device)
        
        for role, state_dict in checkpoint['actors'].items():
            self.actors[role].load_state_dict(state_dict)
        self.critic.load_state_dict(checkpoint['critic'])
        self.inference_net.load_state_dict(checkpoint['inference_net'])
        
        if 'llm' in checkpoint:
            self.llm.model.load_state_dict(checkpoint['llm'])
            
        self.episode = checkpoint.get('episode', 0)
        self.metrics = checkpoint.get('metrics', defaultdict(list))
        
        self.logger.info(f"Loaded checkpoint from {path}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Training Script
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main training function with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SHPPO on Code Contests")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate") 
    parser.add_argument("--lr_llm", type=float, default=1e-5, help="LLM learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="shppo-code-contests", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--small_model", action="store_true", help="Use smaller model for limited GPU memory")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU usage")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with reduced episodes and increased logging")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SHPPOConfig()
    
    # Update config from args
    config.n_episodes = args.episodes
    config.lr_actor = args.lr_actor
    config.lr_critic = args.lr_critic
    config.lr_llm = args.lr_llm
    config.seed = args.seed
    config.wandb_project = args.wandb_project
    config.wandb_run_name = args.wandb_name
    
    if args.device != "auto":
        config.device = args.device
    
    # Debug mode adjustments
    if args.debug:
        config.n_episodes = 50
        config.eval_interval = 10
        config.save_interval = 25
        config.log_interval = 5
        print("🐛 Debug mode: reduced episodes and increased logging frequency")
    
    # Print configuration
    print("🚀 Starting SHPPO Training")
    print("=" * 50)
    print(f"📊 Episodes: {config.n_episodes}")
    print(f"🎭 Roles: {list(config.roles.keys())}")
    print(f"💻 Device: {config.device}")
    print(f"🎯 Learning Rates: Actor={config.lr_actor}, Critic={config.lr_critic}, LLM={config.lr_llm}")
    print(f"📈 Wandb Project: {config.wandb_project}")
    print(f"📁 Checkpoint dir: {config.checkpoint_dir}")
    print("=" * 50)
    print()
    
    # Create trainer (wandb will be initialized in trainer)
    trainer = SHPPOTrainer(config)
    
    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        print(f"📂 Resumed from checkpoint: {args.checkpoint}")
    
    # Start training
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Log final summary
        final_metrics = {}
        for key, values in trainer.metrics.items():
            if values:
                final_metrics[f"final/{key}"] = values[-1]
                final_metrics[f"mean/{key}"] = sum(values) / len(values)
        
        wandb.log(final_metrics)
        print(f"📊 Final average reward: {final_metrics.get('mean/episode_reward', 0):.3f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer._save_checkpoint("interrupted")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        trainer._save_checkpoint("error")
        
    finally:
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    main()