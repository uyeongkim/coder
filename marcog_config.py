# marcog_config.py

from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple, Dict
import math
import torch
# 각 액션 템플릿에 대한 메타데이터 정의
# uses_llm: LLM 생성이 필요한 템플릿인지
# sample_count: uses_llm=True일 경우 생성할 LLM 후보 개수
# is_fixed_response: LLM 호출 없이 고정된 응답(예: 빈 문자열)을 사용하는지
ACTION_METADATA = {
    "plan-subgoal":    {"uses_llm": True, "sample_count": 4, "is_fixed_response": False},
    "generate-code":   {"uses_llm": True, "sample_count": 4, "is_fixed_response": False},
    "fix-code":        {"uses_llm": True, "sample_count": 4, "is_fixed_response": False},
    "noop":            {"uses_llm": False, "sample_count": 1, "is_fixed_response": True}, # noop은 LLM 호출 없고, 빈 응답 하나
    # "rephrase_prompt": {"uses_llm": True, "sample_count": 4, "is_fixed_response": False}, # 플래너에 추가하고 싶다면
}

# 기존 ACTION_TEMPLATES는 ActorNet의 순서를 위해 유지
ACTION_TEMPLATES: Tuple[str, ...] = tuple(ACTION_METADATA.keys()) # 이제 메타데이터에서 키를 가져옴
ACTION_TO_IDX: Dict[str, int] = {action_name: i for i, action_name in enumerate(ACTION_TEMPLATES)}
IDX_TO_ACTION: Dict[int, str] = {i: action_name for i, action_name in enumerate(ACTION_TEMPLATES)}


# 각 역할별로 유효한 액션 인덱스와 해당 액션의 총 샘플 공간 크기 계산
def _calculate_role_action_space_size(action_indices: set[int]) -> int:
    total_size = 0
    # Iterate in a defined order (e.g., sorted by global action index) for consistency
    for idx in sorted(list(action_indices)): # MODIFIED HERE
        action_name = IDX_TO_ACTION[idx]
        total_size += ACTION_METADATA[action_name]["sample_count"]
    return total_size

PLANNER_ACTION_INDICES: set[int] = { ACTION_TO_IDX["plan-subgoal"], ACTION_TO_IDX["noop"], }
CODER_ACTION_INDICES: set[int] = { ACTION_TO_IDX["generate-code"], ACTION_TO_IDX["noop"], }
DEBUGGER_ACTION_INDICES: set[int] = { ACTION_TO_IDX["fix-code"], ACTION_TO_IDX["noop"], }


@dataclass
class BaseRoleConfig:
    role_name: str = ""
    num_agents_in_role: int = 0
    obs_embed_dim: int = 0

    N_ACTION_TEMPLATES: int = 0 # Number of valid action templates for this role
    role_action_space_size: int = field(init=False, repr=False) # Total number of choices in the flattened action space for this role
    def __post_init__(self):
        
        if self.role_name == "planner":
            self.role_action_space_size = _calculate_role_action_space_size(PLANNER_ACTION_INDICES)
        elif self.role_name == "coder":
            self.role_action_space_size = _calculate_role_action_space_size(CODER_ACTION_INDICES)
        elif self.role_name == "debugger":
            self.role_action_space_size = _calculate_role_action_space_size(DEBUGGER_ACTION_INDICES)
        else:
            if self.role_name:
                 raise ValueError(f"Unknown role name: {self.role_name}")
            self.role_action_space_size = 0


@dataclass
class PlannerConfig(BaseRoleConfig):
    role_name: str = "planner"
    num_agents_in_role: int = 3            
    obs_embed_dim: int = 768               
    N_ACTION_TEMPLATES: int = len(PLANNER_ACTION_INDICES)

@dataclass
class CoderConfig(BaseRoleConfig):
    role_name: str = "coder"
    num_agents_in_role: int = 3            
    obs_embed_dim: int = 768               
    N_ACTION_TEMPLATES: int = len(CODER_ACTION_INDICES) 

@dataclass
class DebuggerConfig(BaseRoleConfig):
    role_name: str = "debugger"
    num_agents_in_role: int = 3            
    obs_embed_dim: int = 768               
    N_ACTION_TEMPLATES: int = len(DEBUGGER_ACTION_INDICES) 


@dataclass
class GlobalSHPPOConfig:
    model_dtype: torch.dtype = torch.bfloat16
    llm_model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_dropout: float = 0.05
    llm_actual_hidden_size: Optional[int] = None

    total_planner_agents: int = 3
    total_coder_agents: int = 3
    total_debugger_agents: int = 2

    total_agents_in_pipeline: int = field(init=False)
    lr_actor : float = 5e-4         
    lr_critic: float = 5e-4
    lr_latent: float = 5e-4
    lr_infer : float = 5e-3
    lambda_V_I_guidance_for_latent = 1.0
    lambda_inf_mse =1
    num_minibatches: int = 1  
    gamma: float = 0.95
    lam: float = 0.95
    updates: int = 1000
    epochs: int = 4
    num_problems_per_batch: int = 1
    
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef_actor: float = 0.005
    max_grad_norm: float = 0.3

    lambda_entropy_latent: float = 0.01
    lambda_diversity_latent: float = 0.1

    init_lr: float = 1e-4
    final_lr: float = 5e-6
    lr_schedule: str = "linear"
    warmup_updates: int = 8

    actor_rnn_hidden_dim: int = 64
    latent_dim: int = 3
    mlp_hidden_dim: int = 64
    
    hete_layer_input_dim: int = 64 
    hete_layer_output_dim: int = 64
    actor_final_mlp_output_dim: int = 64

    global_state_dim: int = 768 
    num_scalar_global_features: int = 2

    critic_rnn_hidden_dim: int = 64
    
    max_steps_per_episode: int = 100
    
    use_gae_normalization: bool = True
    value_loss_clipping: bool = True
    value_clip_range: float = 0.1

    wandb_project_name: str = "SHPPO"
    wandb_run_name_prefix: str = "run"
    log_interval: int = 1
    evaluate_interval: int = 5

    device: Optional[Any] = None

    planner_role_config: Optional[PlannerConfig] = field(default=None)
    coder_role_config: Optional[CoderConfig] = field(default=None)
    debugger_role_config: Optional[DebuggerConfig] = field(default=None)

    def __post_init__(self):
        self.total_agents_in_pipeline = self.total_planner_agents + \
                                         self.total_coder_agents + \
                                         self.total_debugger_agents
        
        temp_llm_hidden_size_for_init = self.llm_actual_hidden_size if self.llm_actual_hidden_size is not None else 768 
        temp_obs_embed_dim = temp_llm_hidden_size_for_init + self.num_scalar_global_features 
        
        self.planner_role_config = PlannerConfig(num_agents_in_role=self.total_planner_agents, obs_embed_dim=temp_obs_embed_dim)
        self.coder_role_config = CoderConfig(num_agents_in_role=self.total_coder_agents, obs_embed_dim=temp_obs_embed_dim)
        self.debugger_role_config = DebuggerConfig(num_agents_in_role=self.total_debugger_agents, obs_embed_dim=temp_obs_embed_dim)

        self.global_state_dim = (temp_llm_hidden_size_for_init * 2) + self.num_scalar_global_features


    def update_llm_dims(self, llm_hidden_size: int, llm_compute_dtype: torch.dtype): 
        import logging
        logger = logging.getLogger(__name__) 
        self.llm_actual_hidden_size = llm_hidden_size
        self.model_dtype = llm_compute_dtype 
        
        self.global_state_dim = (self.llm_actual_hidden_size * 2) + self.num_scalar_global_features
        
        new_obs_embed_dim = self.llm_actual_hidden_size + self.num_scalar_global_features
        if self.planner_role_config:
            self.planner_role_config.obs_embed_dim = new_obs_embed_dim
        if self.coder_role_config:
            self.coder_role_config.obs_embed_dim = new_obs_embed_dim
        if self.debugger_role_config:
            self.debugger_role_config.obs_embed_dim = new_obs_embed_dim
            
        logger.info(f"Config updated: llm_actual_hidden_size={self.llm_actual_hidden_size}, "
                    f"model_dtype={self.model_dtype}, "
                    f"global_state_dim={self.global_state_dim}, agent_obs_embed_dim={new_obs_embed_dim}")

    def get_learning_rate(self, current_update: int) -> float:
        if self.updates <= self.warmup_updates: 
            if self.warmup_updates > 0 and current_update < self.warmup_updates :
                 return self.init_lr * (current_update / self.warmup_updates)
            return self.init_lr


        if current_update < self.warmup_updates:
            return self.init_lr * (current_update / self.warmup_updates)
        
        progress = (current_update - self.warmup_updates) / (self.updates - self.warmup_updates)
        progress = min(1.0, max(0.0, progress))
        
        if self.lr_schedule == "cosine":
            return self.final_lr + 0.5 * (self.init_lr - self.final_lr) * (1 + math.cos(math.pi * progress))
        elif self.lr_schedule == "linear":
            return self.init_lr - progress * (self.init_lr - self.final_lr)
        elif self.lr_schedule == "constant":
            return self.init_lr
        else:
            raise ValueError(f"Unknown learning rate schedule: {self.lr_schedule}")