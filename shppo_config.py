# shppo_config.py
import math
from dataclasses import dataclass, field
from typing import List, Tuple

ACTION_TEMPLATES_TUPLE: Tuple[str, ...] = (
    "plan-subgoal", "rephrase-prompt", "assess-subgoals", "generate-code",
    "optimize-code", "self-review", "patch-bug", "unit-fix", "noop",
)
N_ACTION_TEMPLATES_CONST = len(ACTION_TEMPLATES_TUPLE)

@dataclass
class SHPPOConfig:
    """Configuration settings for the SHPPO agent and training environment."""
    ACTION_TEMPLATES: Tuple[str, ...] = ACTION_TEMPLATES_TUPLE
    N_ACTION_TEMPLATES: int = N_ACTION_TEMPLATES_CONST # Number of roles/action types an agent can choose

    # MARL Settings
    num_marl_agents: int = 3 # Number of MARL agents acting sequentially within one team step

    # General RL Settings
    seed: int = 42
    total_timesteps: int = 1000 # Total team environment steps for training
    num_envs: int = 2             # Number of parallel problem environments
    num_steps_per_env: int = 10   # Number of "team steps" per rollout for each parallel environment

    # PPO Update Hyperparameters
    num_minibatches: int = 1
    ppo_epochs: int = 4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    norm_adv: bool = True
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01 # Entropy bonus for the PPO actor (role selection policy)

    # Optimizer Learning Rates
    actor_learning_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    latent_learning_rate: float = 5e-4
    inference_learning_rate: float = 5e-3
    llm_lora_learning_rate: float = 5e-5

    # SHPPO Specific Loss Coefficients (as described in SHPPO paper e.g. Eq. 10)
    lambda_e_latent: float = 0.01  # For LatentNet: entropy of latent distributions (L_e)
    lambda_d_latent: float = 0.1   # For LatentNet: diversity of latent variables (L_d)
    # Note: lambda_V_I_mse is not directly used as an MSE factor for LatentNet loss with V_I.
    # Instead, V_I is maximized, and InferenceNet has its own MSE loss (lambda_inf_mse).
    # See Eq. (6) and Eq. (10) for LatentNet, Eq. (11) for InferenceNet.
    lambda_V_I_guidance_for_latent: float = 1.0 # Implicitly used as LatentNet tries to maximize V_I (coefficient is -1 in loss Eq. 10)
    lambda_inf_mse: float = 1.0 # For InferenceNet: supervised loss against actual returns (Eq. 11)

    adam_eps: float = 1e-5
    weight_decay: float = 0.0
    optimizer_type: str = "Adam"

    # LLM and LoRA Configuration
    llm_model_name: str = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int4"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Network Dimensions
    obs_embed_dim: int = 64       # Local observation embedding dim for each MARL agent
    actor_rnn_hidden_dim: int = 64 # Actor RNN hidden dim for each MARL agent
    latent_dim: int = 3           # Latent variable dim for each role (ACTION_TEMPLATE) per MARL agent
    mlp_hidden_dim: int = 64      # General MLP hidden dimension

    hete_layer_input_dim: int = actor_rnn_hidden_dim  # Input dimension for the HeteLayer
    hete_layer_output_dim: int = 64 # Output dimension for the HeteLayer
    actor_final_mlp_output_dim: int = 64 # Dimension before the policy head in ActorNet
    
    global_state_dim_for_critic: int = 128    # Input dim for centralized CriticNet
    global_state_dim_for_inference: int = 128 # Input dim for centralized InferenceNet
    critic_rnn_hidden_dim: int = 64           # Centralized critic RNN hidden dim

    # Evaluation and Logging
    evaluate_interval: int = 25 # PPO update cycles
    evaluate_episodes: int = 10
    wandb_project_name: str = "SequentialMARL_SHPPO_CodeGen"
    wandb_run_name: str = "seq_marl_shppo_default_run"

    # Episode and Generation Settings
    max_team_episode_steps: int = 10 # Max "team steps" per episode.
    max_llm_input_length: int = 1024
    max_llm_new_tokens: int = 256
    max_prompt_length_for_embedding: int = 256

    max_grad_norm: float = 0.5

    # Internal dynamic settings, will be updated during initialization
    llm_actual_hidden_size: int = 0
    obs_simple_code_embed_dim: int = 20
    obs_simple_plan_embed_dim: int = 20
    obs_simple_error_embed_dim: int = 20
    state_dim_before_projection: int = 0 # For local observation projection of each MARL agent

    # Dataset loading settings
    dataset_max_problems: int = 20
    dataset_max_cases: int = 3
    execution_results_csv_path: str = "marl_execution_results.csv"

    # Calculated after initialization
    rollout_buffer_size_team_steps: int = 0
    minibatch_size_team_steps: int = 0


    def __post_init__(self):
        """Calculate derived configuration values."""
        self.rollout_buffer_size_team_steps = self.num_envs * self.num_steps_per_env
        if self.num_minibatches > 0:
            if self.rollout_buffer_size_team_steps % self.num_minibatches != 0:
                raise ValueError("rollout_buffer_size_team_steps must be divisible by num_minibatches")
            self.minibatch_size_team_steps = self.rollout_buffer_size_team_steps // self.num_minibatches
        else: # Treat as num_minibatches = 1 if 0 or negative
            self.minibatch_size_team_steps = self.rollout_buffer_size_team_steps
            self.num_minibatches = 1
        
        if self.hete_layer_input_dim != self.actor_rnn_hidden_dim:
            # As per SHPPO paper Figure 2(c), HeteLayer input is ActorNet RNN's output.
            raise ValueError("hete_layer_input_dim should be equal to actor_rnn_hidden_dim")