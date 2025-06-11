# shppo.py
import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple, Optional
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import logging
import wandb # itertools는 이전 코드 스니펫에 있었으나 현재 코드에서는 직접 사용되지 않으므로 제거했습니다.

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    GenerationConfig
)
from peft import get_peft_model, LoraConfig, PeftModel

from shppo_config import SHPPOConfig # Assuming this file exists and SHPPOConfig is defined
from shppo_env import SHPPOCodeEnv, CodeContestDataset # Assuming these exist

logger = logging.getLogger(__name__)

#######
def ortho_init(m: nn.Module, gain: float = 1.0):
    """Orthogonal initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MLPBlock(nn.Module):
    """A simple MLP block with configurable layers and ReLU activations."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        layers_list = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers_list.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            current_dim = hidden_dim
        layers_list.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers_list)
        self.net.apply(lambda m: ortho_init(m, math.sqrt(2)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Encoder(nn.Module):
    """
    Encoder module for the LatentNet. (Fig. 2a in SHPPO paper).
    Processes agent's local observation and actor's previous hidden state
    to output parameters (mu, sigma_raw) for role-specific latent distributions.
    """
    def __init__(self, config: SHPPOConfig):
        super().__init__()
        input_dim = config.obs_embed_dim + config.actor_rnn_hidden_dim
        self.encoder_mlp = MLPBlock(input_dim, config.mlp_hidden_dim, hidden_dim=config.mlp_hidden_dim, num_layers=3)
        self.fc_mu = nn.Linear(config.mlp_hidden_dim, config.N_ACTION_TEMPLATES * config.latent_dim)
        self.fc_sigma = nn.Linear(config.mlp_hidden_dim, config.N_ACTION_TEMPLATES * config.latent_dim)
        self.fc_mu.apply(lambda m: ortho_init(m, 0.01))
        self.fc_sigma.apply(lambda m: ortho_init(m, 0.01))
    
    def forward(self, obs_emb: torch.Tensor, h_actor_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_emb, h_actor_prev], dim=-1)
        x = self.encoder_mlp(x)
        return self.fc_mu(x), self.fc_sigma(x)

class LatentNet(nn.Module):
    """
    Latent Network (LatentNet) (Fig. 2a in SHPPO paper).
    Generates sampled latent variables (z) for roles using an Encoder,
    along with the mean (mu) and standard deviation (sigma) of the learned distributions.
    """
    def __init__(self, encoder: Encoder, config: SHPPOConfig):
        super().__init__()
        self.encoder = encoder
        self.N_ACTION_TEMPLATES = config.N_ACTION_TEMPLATES
        self.latent_dim = config.latent_dim
    
    def forward(self, obs_emb: torch.Tensor, h_actor_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_flat, sigma_raw_flat = self.encoder(obs_emb, h_actor_prev)
        
        mu = mu_flat.view(-1, self.N_ACTION_TEMPLATES, self.latent_dim)
        sigma_raw = sigma_raw_flat.view(-1, self.N_ACTION_TEMPLATES, self.latent_dim)
        
        sigma = F.softplus(sigma_raw) + 1e-5
        
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return z, mu, sigma

class InferenceNet(nn.Module):
    """
    Inference Network (InferenceNet) (Fig. 2b in SHPPO paper).
    Centralized network taking global state and all agents' latent distribution parameters (mu, sigma)
    to predict an intrinsic value V_I, guiding LatentNet learning. Trained against actual returns (Eq. 11).
    """
    def __init__(self, config: SHPPOConfig):
        super().__init__()
        input_dim = config.global_state_dim_for_inference + \
                      (2 * config.num_marl_agents * config.N_ACTION_TEMPLATES * config.latent_dim)
        self.v_head = MLPBlock(input_dim, 1, hidden_dim=config.mlp_hidden_dim, num_layers=3)
    
    def forward(self, glob_s_emb: torch.Tensor, all_mu_roles_all_agents: torch.Tensor, all_sig_roles_all_agents: torch.Tensor) -> torch.Tensor:
        batch_size = glob_s_emb.shape[0]
        
        mu_flat = all_mu_roles_all_agents.reshape(batch_size, -1)
        sig_flat = all_sig_roles_all_agents.reshape(batch_size, -1)
        
        x = torch.cat([glob_s_emb, mu_flat, sig_flat], dim=-1)
        return self.v_head(x).squeeze(-1)

class HeteLayerDecoder(nn.Module):
    """
    Decoder for the Heterogeneous Layer (HeteLayer) (Fig. 2d in SHPPO paper).
    Decodes a role-specific latent variable 'z_role' (l_i in paper) into
    weights (W_i) and biases (b_i) for that role's HeteLayer in ActorNet.
    """
    def __init__(self, latent_dim: int, hete_input_dim: int, hete_output_dim: int):
        super().__init__()
        self.w_decoder = nn.Linear(latent_dim, hete_input_dim * hete_output_dim)
        self.b_decoder = nn.Linear(latent_dim, hete_output_dim)
        self.hete_input_dim = hete_input_dim
        self.hete_output_dim = hete_output_dim
        
        self.w_decoder.apply(lambda m: ortho_init(m, math.sqrt(0.1))) 
        self.b_decoder.apply(lambda m: ortho_init(m, math.sqrt(0.1)))
    
    def forward(self, z_role: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.w_decoder(z_role).view(-1, self.hete_output_dim, self.hete_input_dim)
        biases = self.b_decoder(z_role)
        return weights, biases

class ActorNet(nn.Module):
    """
    Actor Network (ActorNet) with a Heterogeneous Layer (HeteLayer) (Fig. 2c in SHPPO paper).
    Processes local observations, uses an RNN for memory, and applies role-specific HeteLayers
    (parameters generated by HeteLayerDecoder from latents) to produce action logits for role selection.
    """
    def __init__(self, config: SHPPOConfig):
        super().__init__()
        self.config = config
        self.obs_encoder = MLPBlock(config.obs_embed_dim, config.actor_rnn_hidden_dim, hidden_dim=config.mlp_hidden_dim)
        self.rnn = nn.GRU(config.actor_rnn_hidden_dim, config.actor_rnn_hidden_dim, batch_first=True)
        self.hete_layer_decoder = HeteLayerDecoder(config.latent_dim, config.hete_layer_input_dim, config.hete_layer_output_dim)
        self.final_mlp = MLPBlock(config.hete_layer_output_dim, config.actor_final_mlp_output_dim, hidden_dim=config.mlp_hidden_dim)
        self.policy_head = nn.Linear(config.actor_final_mlp_output_dim, 1)
        self.policy_head.apply(lambda m: ortho_init(m, 0.01))

    def forward(self, agent_obs_emb: torch.Tensor, agent_h_prev: torch.Tensor, z_all_roles_for_this_agent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_agent_size = agent_obs_emb.size(0)
        num_roles = self.config.N_ACTION_TEMPLATES

        obs_features = self.obs_encoder(agent_obs_emb)
        
        rnn_output, h_next_gru_unsq = self.rnn(obs_features.unsqueeze(1), agent_h_prev.unsqueeze(0))
        actor_hidden_state = rnn_output.squeeze(1) 
        
        actor_hidden_state_expanded = actor_hidden_state.unsqueeze(1).repeat(1, num_roles, 1)
        
        if actor_hidden_state_expanded.shape[-1] != self.config.hete_layer_input_dim:
            logger.warning(f"Actor RNN hidden dim ({actor_hidden_state_expanded.shape[-1]}) "
                           f"does not match HeteLayer input dim ({self.config.hete_layer_input_dim}).")

        actor_hidden_state_flat = actor_hidden_state_expanded.reshape(-1, self.config.hete_layer_input_dim)
        
        z_roles_flat = z_all_roles_for_this_agent.reshape(-1, self.config.latent_dim)
        
        W_roles, b_roles = self.hete_layer_decoder(z_roles_flat)
        
        hete_features = torch.bmm(W_roles, actor_hidden_state_flat.unsqueeze(-1)).squeeze(-1) + b_roles
        
        final_features = self.final_mlp(hete_features)
        role_scores_flat = self.policy_head(final_features)
        
        action_logits = role_scores_flat.view(batch_agent_size, num_roles)
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_logits, h_next_gru_unsq.squeeze(0), action_probs

class CriticNet(nn.Module):
    """
    Centralized Critic Network (CriticNet) (Fig. 2b in SHPPO paper).
    Processes team global state embeddings using an RNN to estimate the team value V_C.
    """
    def __init__(self, config: SHPPOConfig):
        super().__init__()
        self.global_state_projector = MLPBlock(config.global_state_dim_for_critic, config.critic_rnn_hidden_dim, hidden_dim=config.mlp_hidden_dim, num_layers=2)
        self.rnn = nn.GRU(config.critic_rnn_hidden_dim, config.critic_rnn_hidden_dim, batch_first=True)
        self.value_head = MLPBlock(config.critic_rnn_hidden_dim, 1, hidden_dim=config.mlp_hidden_dim)
        self.value_head.apply(lambda m: ortho_init(m, 1.0))
    
    def forward(self, glob_s_emb: torch.Tensor, h_crit_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected_state = self.global_state_projector(glob_s_emb)
        rnn_output, h_crit_next_unsq = self.rnn(projected_state.unsqueeze(1), h_crit_prev.unsqueeze(0))
        value_prediction = self.value_head(rnn_output.squeeze(1)).squeeze(-1)
        return value_prediction, h_crit_next_unsq.squeeze(0)
###
def build_networks(config: SHPPOConfig, device: torch.device) -> Dict[str, Any]:
    """Builds and initializes all networks, tokenizer, and updates config with dynamic LLM dims."""
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True 
    )
    try: 
        llm_base = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name, 
            quantization_config=bnb_cfg, 
            device_map={"":device},
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        )
    except Exception as e: 
        logger.error(f"LLM base model loading failed: {e}. Check model name, network access, and CUDA setup."); raise
        
    lora_cfg = LoraConfig(
        r=config.lora_r, 
        lora_alpha=config.lora_alpha, 
        target_modules=config.lora_target_modules, 
        lora_dropout=config.lora_dropout, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    llm_peft_model = get_peft_model(llm_base, lora_cfg)
    llm_trainable_params = [p for n,p in llm_peft_model.named_parameters() if "lora_" in n and p.requires_grad]
    
    if not llm_trainable_params: 
        logger.warning("No LoRA trainable parameters found in LLM.")
    else: 
        logger.info(f"LLM LoRA applied. Trainable LoRA params: {sum(p.numel() for p in llm_trainable_params)}")
        
    config.llm_actual_hidden_size = llm_base.config.hidden_size 
    
    num_scalar_global_feats = getattr(config, 'num_scalar_global_features', 2)
    config.global_state_dim_for_critic = config.llm_actual_hidden_size + num_scalar_global_feats
    config.global_state_dim_for_inference = config.llm_actual_hidden_size + num_scalar_global_feats
    logger.info(f"Set global_state_dim_for_critic: {config.global_state_dim_for_critic}")
    logger.info(f"Set global_state_dim_for_inference: {config.global_state_dim_for_inference}")
        
    encoder_module = Encoder(config).to(device)
    latent_net_module = LatentNet(encoder_module, config).to(device)
    actor_net_module = ActorNet(config).to(device)
    critic_net_module = CriticNet(config).to(device)
    inference_net_module = InferenceNet(config).to(device)
    
    params_dict = {
        "llm_lora": llm_trainable_params, 
        "actor_core": list(actor_net_module.parameters()), 
        "critic": list(critic_net_module.parameters()), 
        "latent": list(latent_net_module.parameters()), 
        "inference": list(inference_net_module.parameters())
    }
    
    return {
        "llm_model": llm_peft_model, 
        "tokenizer": tokenizer, 
        "actor_net": actor_net_module, 
        "critic_net": critic_net_module, 
        "latent_net": latent_net_module, 
        "inference_net": inference_net_module, 
        "params": params_dict
    }

def cosine_diversity(z_roles_batch: torch.Tensor) -> torch.Tensor:
    """
    Calculates cosine diversity among roles for a batch (Eq. 8 in SHPPO paper). Maximized.
    """
    if z_roles_batch.size(1) <= 1: 
        return torch.tensor(0.0, device=z_roles_batch.device)
    
    batch_diversities = []
    for i in range(z_roles_batch.size(0)): 
        z_item_roles = z_roles_batch[i] 
        if z_item_roles.size(0) <= 1: 
            batch_diversities.append(torch.tensor(0.0, device=z_item_roles.device))
            continue
        
        z_normalized = F.normalize(z_item_roles, p=2, dim=-1) 
        similarity_matrix = torch.matmul(z_normalized, z_normalized.transpose(-2, -1))
        
        mask = torch.triu(torch.ones_like(similarity_matrix, dtype=torch.bool), diagonal=1)
        
        if mask.sum() > 0: 
            distances = 1.0 - similarity_matrix[mask]
            batch_diversities.append(distances.mean())
        else: 
            batch_diversities.append(torch.tensor(0.0, device=z_item_roles.device))
            
    if not batch_diversities:
        return torch.tensor(0.0, device=z_roles_batch.device)
    return torch.stack(batch_diversities).mean()

class SHPPOTrainer:
    def __init__(self, cfg: SHPPOConfig, device: torch.device, env: SHPPOCodeEnv):
        self.cfg = cfg
        self.device = device
        self.env = env
        torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
        
        network_components = build_networks(cfg, device) 
        self.llm_model: PeftModel = network_components["llm_model"]
        self.tokenizer = network_components["tokenizer"]
        self.actor_net: ActorNet = network_components["actor_net"]
        self.critic_net: CriticNet = network_components["critic_net"]
        self.latent_net: LatentNet = network_components["latent_net"]
        self.inference_net: InferenceNet = network_components["inference_net"]
        
        all_params = network_components["params"]
        
        self.cfg.state_dim_before_projection = (
            self.cfg.llm_actual_hidden_size + 
            self.cfg.obs_simple_plan_embed_dim +
            self.cfg.obs_simple_code_embed_dim + 
            self.cfg.obs_simple_error_embed_dim + 
            self.cfg.N_ACTION_TEMPLATES 
        )
        
        if self.cfg.state_dim_before_projection != self.cfg.obs_embed_dim:
            self.state_projection_layer: nn.Module = nn.Linear(
                self.cfg.state_dim_before_projection, self.cfg.obs_embed_dim
            ).to(self.device)
            ortho_init(self.state_projection_layer, gain=math.sqrt(2))
        else:
            self.state_projection_layer = nn.Identity()
            
        actor_trainable_params = list(all_params["actor_core"])
        if not isinstance(self.state_projection_layer, nn.Identity):
            actor_trainable_params += list(self.state_projection_layer.parameters())
        
        self.opt_actor = getattr(optim, cfg.optimizer_type)(actor_trainable_params, lr=cfg.actor_learning_rate, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
        self.opt_critic = getattr(optim, cfg.optimizer_type)(all_params["critic"], lr=cfg.critic_learning_rate, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
        self.opt_latent = getattr(optim, cfg.optimizer_type)(all_params["latent"], lr=cfg.latent_learning_rate, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
        self.opt_inference = getattr(optim, cfg.optimizer_type)(all_params["inference"], lr=cfg.inference_learning_rate, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
        
        if all_params["llm_lora"]:
            self.opt_llm_lora = getattr(optim, cfg.optimizer_type)(all_params["llm_lora"], lr=cfg.llm_lora_learning_rate, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
        else:
            self.opt_llm_lora = None
            
        S, B, Na, Nr, Dl = cfg.num_steps_per_env, cfg.num_envs, cfg.num_marl_agents, cfg.N_ACTION_TEMPLATES, cfg.latent_dim
        Da_loc, Dglob_c, Dglob_i = cfg.obs_embed_dim, cfg.global_state_dim_for_critic, cfg.global_state_dim_for_inference
        Drnn_a, Drnn_c = cfg.actor_rnn_hidden_dim, cfg.critic_rnn_hidden_dim
        
        self.rollout_buffer = {
            "local_obs_embeddings": torch.zeros((S, B, Na, Da_loc), dtype=torch.float32, device=device),
            "global_state_embeddings_critic": torch.zeros((S, B, Dglob_c), dtype=torch.float32, device=device),
            "global_state_embeddings_inference": torch.zeros((S, B, Dglob_i), dtype=torch.float32, device=device),
            "actor_hidden_states": torch.zeros((S, B, Na, Drnn_a), dtype=torch.float32, device=device),
            "critic_hidden_states": torch.zeros((S, B, Drnn_c), dtype=torch.float32, device=device),
            "latents_z_all_roles": torch.zeros((S, B, Na, Nr, Dl), dtype=torch.float32, device=device),
            "latents_mu_all_roles": torch.zeros((S, B, Na, Nr, Dl), dtype=torch.float32, device=device),
            "latents_sigma_all_roles": torch.zeros((S, B, Na, Nr, Dl), dtype=torch.float32, device=device),
            "actions": torch.zeros((S, B, Na), dtype=torch.long, device=device),
            "log_probs": torch.zeros((S, B, Na), dtype=torch.float32, device=device),
            "team_rewards": torch.zeros((S, B), dtype=torch.float32, device=device),
            "team_values": torch.zeros((S, B), dtype=torch.float32, device=device),
            "team_dones": torch.zeros((S, B), dtype=torch.bool, device=device),
        }
        self.rollout_problem_llm_responses_for_csv: List[Tuple[Optional[Dict[str, Any]], str]] = [(None, "") for _ in range(cfg.num_envs)]
        logger.info("SHPPOTrainer initialized.")

    def _simple_embed(self, text: str, dim: int) -> np.ndarray:
        vec=np.zeros(dim,dtype=np.float32)
        if text and dim>0:
            processed_text=text[:dim*4] 
            for char_code in [ord(c) for c in processed_text if ord(c) < 256]:
                vec[char_code % dim] += 1
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 1e-9 else vec
        return vec

    def _get_llm_embedding(self, text: str, max_length: int, track_grads: bool = False) -> torch.Tensor:
        """Generates embedding, optionally tracking gradients for LLM LoRA."""
        if not text: 
            return torch.zeros(self.cfg.llm_actual_hidden_size, device=self.device)
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length").to(self.device)
        
        context_manager = torch.enable_grad() if track_grads else torch.no_grad()
        with context_manager:
            if hasattr(self.llm_model, "get_input_embeddings"):
                base_model_for_embeddings = self.llm_model.get_input_embeddings()
            elif hasattr(self.llm_model, "base_model") and \
                 hasattr(self.llm_model.base_model, "model") and \
                 hasattr(self.llm_model.base_model.model, "get_input_embeddings"):
                base_model_for_embeddings = self.llm_model.base_model.model.get_input_embeddings()
            else:
                try:
                    base_model_for_embeddings = self.llm_model.get_input_embeddings() 
                except AttributeError:
                    raise AttributeError("LLM model does not have a recognizable get_input_embeddings method.")

            embeddings = base_model_for_embeddings(inputs.input_ids)
            attention_mask_expanded = inputs.attention_mask.unsqueeze(-1).expand_as(embeddings).float()
            sum_embeddings = torch.sum(embeddings * attention_mask_expanded, dim=1)
            num_valid_tokens = torch.clamp(inputs.attention_mask.sum(dim=1), min=1e-9)
            pooled_embedding = (sum_embeddings / num_valid_tokens.unsqueeze(-1)).squeeze(0)
        
        if not track_grads and pooled_embedding.grad_fn is not None: # Ensure detachment if grads were not intended
            return pooled_embedding.detach()
        return pooled_embedding

    def get_agent_observation_embedding(self, agent_local_state_components: Dict[str, Any]) -> torch.Tensor:
        prompt_text = agent_local_state_components.get("prompt", "")
        team_code_text = agent_local_state_components.get("team_overall_code", "")
        team_plan_text = agent_local_state_components.get("team_plan", "") 
        team_pass_fraction_val = agent_local_state_components.get("team_pass_fraction", 0.0)
        team_error_text = agent_local_state_components.get("team_errors_summary", "")
        my_last_action_str = agent_local_state_components.get("my_last_action_str", self.cfg.ACTION_TEMPLATES[-1])
        
        # Determine if LoRA weights should be trained by this observation's use (typically for actor)
        track_llm_grads = self.opt_llm_lora is not None and (self.actor_net.training or self.llm_model.training)

        prompt_emb = self._get_llm_embedding(prompt_text, self.cfg.max_prompt_length_for_embedding, track_grads=track_llm_grads)
        
        code_emb = torch.tensor(self._simple_embed(team_code_text, self.cfg.obs_simple_code_embed_dim), dtype=torch.float32, device=self.device)
        plan_emb = torch.tensor(self._simple_embed(team_plan_text, self.cfg.obs_simple_plan_embed_dim), dtype=torch.float32, device=self.device)
        
        error_emb_np = np.zeros(self.cfg.obs_simple_error_embed_dim, dtype=np.float32)
        if self.cfg.obs_simple_error_embed_dim > 0:
            error_emb_np[0] = team_pass_fraction_val
            error_text_dim = self.cfg.obs_simple_error_embed_dim - 1
            if error_text_dim > 0:
                error_emb_np[1:] = self._simple_embed(team_error_text, error_text_dim)
        error_emb = torch.tensor(error_emb_np, dtype=torch.float32, device=self.device)
        
        last_action_idx = self.cfg.ACTION_TEMPLATES.index(my_last_action_str) if my_last_action_str in self.cfg.ACTION_TEMPLATES else self.cfg.N_ACTION_TEMPLATES - 1
        last_action_one_hot = F.one_hot(torch.tensor(last_action_idx, device=self.device, dtype=torch.long), num_classes=self.cfg.N_ACTION_TEMPLATES).float()
            
        combined_features = torch.cat([prompt_emb, plan_emb, code_emb , error_emb, last_action_one_hot], dim=-1)
        
        return self.state_projection_layer(combined_features)
        
    def get_team_global_state_embedding(self, team_global_state_components: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Constructs global state embeddings using the problem prompt, no LLM grads tracked here."""
        prompt_text = team_global_state_components.get("prompt", "") 
        prompt_emb = self._get_llm_embedding(prompt_text, self.cfg.max_prompt_length_for_global_embedding, track_grads=False)

        pass_fraction = team_global_state_components.get("team_pass_fraction", 0.0)
        episode_prog = team_global_state_components.get("episode_steps", 0)
        normalized_eps = episode_prog / self.cfg.max_team_episode_steps if self.cfg.max_team_episode_steps > 0 else 0.0
        
        scalar_features_list = []
        if hasattr(self.cfg, 'num_scalar_global_features') and self.cfg.num_scalar_global_features > 0:
            scalar_features_list.append(pass_fraction)
            if self.cfg.num_scalar_global_features > 1:
                 scalar_features_list.append(normalized_eps)

        if scalar_features_list:
            scalar_features = torch.tensor(scalar_features_list, dtype=torch.float32, device=self.device)
            if scalar_features.shape[0] != self.cfg.num_scalar_global_features:
                if scalar_features.shape[0] < self.cfg.num_scalar_global_features:
                    padding = torch.zeros(self.cfg.num_scalar_global_features - scalar_features.shape[0], device=self.device)
                    scalar_features = torch.cat([scalar_features, padding], dim=0)
                else:
                    scalar_features = scalar_features[:self.cfg.num_scalar_global_features]
            combined_global_emb = torch.cat([prompt_emb, scalar_features], dim=-1)
        else:
            combined_global_emb = prompt_emb

        glob_s_critic = torch.zeros(self.cfg.global_state_dim_for_critic, device=self.device)
        len_to_copy_critic = min(combined_global_emb.shape[0], self.cfg.global_state_dim_for_critic)
        glob_s_critic[:len_to_copy_critic] = combined_global_emb[:len_to_copy_critic]

        glob_s_inference = torch.zeros(self.cfg.global_state_dim_for_inference, device=self.device)
        len_to_copy_inference = min(combined_global_emb.shape[0], self.cfg.global_state_dim_for_inference)
        glob_s_inference[:len_to_copy_inference] = combined_global_emb[:len_to_copy_inference]
        
        return glob_s_critic, glob_s_inference

    @torch.no_grad()
    def generate_llm_response_for_action(
        self,
        agent_local_obs_components: Dict[str, Any],
        action_template: str
    ) -> str:
        """Generates a detailed prompt based on the action template and agent's local observation, then calls the LLM."""
        prompt_problem = agent_local_obs_components.get('prompt', "")
        prev_code = agent_local_obs_components.get('team_overall_code', "")
        prev_plan = agent_local_obs_components.get('team_plan', "")
        errs = agent_local_obs_components.get('team_errors_summary', "")
        pf = agent_local_obs_components.get('team_pass_fraction', 0.0)
        marl_agent_idx = agent_local_obs_components.get('my_id', -1)

        body = ""
        if action_template == "plan-subgoal":
            body = (
                f"Task:\n{prompt_problem}\n\n"
                f"Previous plan:\n{prev_plan}\n\n"
                f"Current Pass fraction: {pf*100:.1f}%\n"
                f"Observed Errors:\n{errs}\n\n"
                f"Your Role (Agent {marl_agent_idx}): {action_template}\n"
                f"Generate a concise, actionable, step-by-step plan (subgoals) to solve the task. Focus on the next few critical steps."
            )
        elif action_template == "rephrase-prompt":
            body = (
                f"Original Prompt:\n{prompt_problem}\n\n"
                f"Your Role (Agent {marl_agent_idx}): {action_template}\n"
                f"Rephrase the original prompt to be clearer, more specific, and highlight key constraints or objectives. This will help guide code generation."
            )
        elif action_template == "assess-subgoals":
            if not prev_plan.strip(): return "# Skipped assess-subgoals: No current plan available."
            body = (
                f"Task:\n{prompt_problem}\n\n"
                f"Current Plan (Subgoals) to Assess:\n{prev_plan}\n"
                f"Current Pass fraction: {pf*100:.1f}%\n"
                f"Observed Errors:\n{errs}\n\n"
                f"Your Role (Agent {marl_agent_idx}): {action_template}\n"
                f"Critically assess the provided plan. Is it logical? Does it cover all aspects of the task? Are the subgoals achievable and well-defined? Suggest specific improvements or point out flaws."
            )
        elif action_template == "generate-code":
            body = (
                f"Task Description:\n{prompt_problem}\n\n"
                f"Current Team Plan (Subgoals):\n{prev_plan}\n"
                f"Current Team Pass Fraction: {pf*100:.1f}%\n"
                f"Current Team Errors (if any):\n{errs}\n\n"
                f"Your Role (Agent {marl_agent_idx}): {action_template}\n"
                f"Based on the task and plan, implement the core logic in a Python function `def solve(input_data: str) -> str:`.\n"
                "Constraints:\n"
                "- The function must take a single string `input_data` (representing stdin) and return a string (representing stdout).\n"
                "- Do NOT use `input()` or `print()` statements inside `solve`.\n"
                "- Output ONLY the complete Python code for the `solve` function. No extra text, explanations, or markdown markers like ```python ... ```."
            )
        elif action_template == "optimize-code":
            if not prev_code.strip(): return "# Skipped optimize-code: No code to optimize."
            body = (
                f"Task:\n{prompt_problem}\n\n"
                f"Code to Optimize:\n```python\n{prev_code}\n```\n"
                f"Current Team Plan:\n{prev_plan}\n"
                f"Current Pass fraction: {pf*100:.1f}%\n"
                f"Observed Errors:\n{errs}\n\n"
                f"Your Role (Agent {marl_agent_idx}): {action_template}\n"
                f"Optimize the provided Python code for time/space complexity or readability, ensuring functionality remains identical. Output ONLY the complete optimized `solve` function code."
            )
        elif action_template == "self-review":
            if not prev_code.strip(): return "# Skipped self-review: No code to review."
            body = (
                f"Task:\n{prompt_problem}\n\n"
                f"Code for Review:\n```python\n{prev_code}\n```\n"
                f"Current Pass fraction: {pf*100:.1f}%\n"
                f"Observed Errors:\n{errs}\n\n"
                f"Your Role (Agent {marl_agent_idx}): {action_template}\n"
                f"Review the provided code. Identify potential logical errors, bugs, inefficiencies, or areas not adhering to the task requirements. Provide a concise review with specific, actionable suggestions."
            )
        elif action_template == "patch-bug":
            if not prev_code.strip(): return "# Skipped patch-bug: No code to debug."
            if not errs.strip() or errs.startswith("AllTestsPassed"): return "# Skipped patch-bug: No specific errors to patch or all tests passed."
            body = (
                f"Task:\n{prompt_problem}\n\n"
                f"Code with Bugs:\n```python\n{prev_code}\n```\n"
                f"Observed Errors During Testing:\n{errs}\n\n"
                f"Your Role (Agent {marl_agent_idx}): {action_template}\n"
                f"Fix the bugs in the provided code based on the observed errors. Output ONLY the complete corrected `solve` function code."
            )
        elif action_template == "unit-fix":
            if not prev_code.strip(): return "# Skipped unit-fix: No code to fix."
            if not errs.strip() or errs.startswith("AllTestsPassed"): return "# Skipped unit-fix: No specific errors for unit-fix or all tests passed."
            body = (
                f"Task:\n{prompt_problem}\n\n"
                f"Failing Code:\n```python\n{prev_code}\n```\n"
                f"Current Pass fraction: {pf*100:.1f}%\n"
                f"Test Failures / Errors:\n{errs}\n\n"
                f"Your Role (Agent {marl_agent_idx}): {action_template}\n"
                f"Focus on fixing the specific issues highlighted by the test failures or errors to improve the pass fraction. Output ONLY the corrected `solve` function code."
            )
        elif action_template == "noop":
            return "# No operation selected for this step by agent."
        else: # Fallback for unknown action template
            logger.warning(f"Unknown action_template in generate_llm_response_for_action: {action_template}. Using generic prompt.")
            body = (
                f"Task:\n{prompt_problem}\n\nTeam Code:\n```python\n{prev_code}\n```\nPlan:\n{prev_plan}\nErrors:\n{errs}\nPF: {pf*100:.1f}%\n"
                f"Your Role (Agent {marl_agent_idx}): {action_template}\nYour Response:"
            )

        messages = [
            {"role": "system", "content": "You are a highly skilled AI programming assistant. Follow instructions precisely. If generating code, provide only the raw Python code for the 'solve' function."},
            {"role": "user", "content": body},
        ]
        
        prompt_text_for_llm = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore [union-attr]
        
        responses = self.call_llm_batch([prompt_text_for_llm])
        return responses[0] if responses else f"# LLM call failed for action: {action_template}"
    
    def call_llm_batch(self, prompt_texts: List[str]) -> List[str]:
        """Calls the LLM in batch with a list of prompt texts."""
        if not prompt_texts: return []
        cfg=self.cfg
        
        # Ensure max_input_len is reasonable, considering tokens for generation
        max_input_len = cfg.max_llm_input_length - cfg.max_llm_new_tokens 
        max_input_len = max(50, max_input_len - 20) # Keep a small buffer and minimum length

        inputs=self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True, # Pad to longest in batch
            truncation=True,
            max_length=max_input_len 
        ).to(self.device) # type: ignore [union-attr]
        
        generation_config_dict = {
            "max_new_tokens": cfg.max_llm_new_tokens,
            "do_sample": False, # For deterministic output, can be True for more diverse responses
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id, # type: ignore [union-attr]
            "eos_token_id": self.tokenizer.eos_token_id, # type: ignore [union-attr]
            # "temperature": 0.7, # Example if do_sample=True
            # "top_p": 0.9,       # Example if do_sample=True
        }
        gen_config_obj=GenerationConfig(**generation_config_dict)
        
        # Determine if LLM LoRA parameters are being trained
        is_training_llm_lora = self.opt_llm_lora is not None and self.llm_model.training
        
        # Enable gradients only if LLM LoRA is being trained
        context_manager = torch.enable_grad() if is_training_llm_lora else torch.no_grad()
        
        with context_manager:
            outputs = self.llm_model.generate(**inputs, generation_config=gen_config_obj) # type: ignore [operator]
        
        # Decode generated tokens, excluding input tokens
        decoded_responses = [
            self.tokenizer.decode(outputs[i, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip() # type: ignore [union-attr]
            for i in range(outputs.shape[0])
        ]
        return decoded_responses
    
    def collect_rollouts(self, current_h_actor_teams_in: torch.Tensor, current_h_critic_teams_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        B, Na, Nr, Dl = cfg.num_envs, cfg.num_marl_agents, cfg.N_ACTION_TEMPLATES, cfg.latent_dim
        
        h_actor_teams = current_h_actor_teams_in.detach().clone()
        h_critic_teams = current_h_critic_teams_in.detach().clone()

        all_envs_current_agent_loc_obs_comp, all_envs_current_team_global_state_comp = self.env.reset_all_envs()
        all_envs_current_agent_loc_obs_comp = [obs for obs in all_envs_current_agent_loc_obs_comp]

        self.rollout_problem_llm_responses_for_csv = [(None, "") for _ in range(B)]

        for t_step in range(cfg.num_steps_per_env):
            self.rollout_buffer["actor_hidden_states"][t_step] = h_actor_teams.clone()
            self.rollout_buffer["critic_hidden_states"][t_step] = h_critic_teams.clone()

            current_step_global_s_crit_list = []
            current_step_global_s_inf_list = []
            for env_idx_gs in range(B):
                if all_envs_current_team_global_state_comp[env_idx_gs] is not None:
                    s_c, s_i = self.get_team_global_state_embedding(all_envs_current_team_global_state_comp[env_idx_gs])
                else: 
                    s_c = torch.zeros(self.cfg.global_state_dim_for_critic, device=self.device)
                    s_i = torch.zeros(self.cfg.global_state_dim_for_inference, device=self.device)
                current_step_global_s_crit_list.append(s_c)
                current_step_global_s_inf_list.append(s_i)
            
            stacked_global_s_crit = torch.stack(current_step_global_s_crit_list)
            self.rollout_buffer["global_state_embeddings_critic"][t_step] = stacked_global_s_crit
            self.rollout_buffer["global_state_embeddings_inference"][t_step] = torch.stack(current_step_global_s_inf_list)

            with torch.no_grad(): # Critic is in eval mode during rollouts for value estimation
                team_values_at_t, h_critic_next_for_buffer = self.critic_net(stacked_global_s_crit, h_critic_teams)
            self.rollout_buffer["team_values"][t_step] = team_values_at_t # Already detached if critic_net is under no_grad

            current_t_step_loc_obs_embeddings_list = [torch.zeros(Na, cfg.obs_embed_dim, device=self.device) for _ in range(B)]
            current_t_step_latents_z_list = [torch.zeros(Na, Nr, Dl, device=self.device) for _ in range(B)]
            # ... (initialize other lists for mu, sigma, actions, log_probs)
            current_t_step_latents_mu_list: List[torch.Tensor] = [torch.zeros(Na, Nr, Dl, device=self.device) for _ in range(B)]
            current_t_step_latents_sigma_list: List[torch.Tensor] = [torch.zeros(Na, Nr, Dl, device=self.device) for _ in range(B)]
            current_t_step_actions_list: List[torch.Tensor] = [torch.zeros(Na, dtype=torch.long, device=self.device) for _ in range(B)]
            current_t_step_log_probs_list: List[torch.Tensor] = [torch.zeros(Na, device=self.device) for _ in range(B)]


            next_team_global_states_for_next_iteration = [None for _ in range(B)] 

            for env_idx in range(B):
                if self.env.env_states[env_idx].get("episode_done", False) and t_step > 0 : 
                    self.rollout_buffer["team_rewards"][t_step, env_idx] = 0.0
                    self.rollout_buffer["team_dones"][t_step, env_idx] = True
                    if self.rollout_problem_llm_responses_for_csv[env_idx][0] is None:
                        self.rollout_problem_llm_responses_for_csv[env_idx] = (self.env.env_states[env_idx].get('task_data',{}), "#DoneEarlier")
                    next_team_global_states_for_next_iteration[env_idx] = all_envs_current_team_global_state_comp[env_idx] 
                    continue

                for agent_idx_turn in range(Na):
                    current_agent_loc_obs_comp = all_envs_current_agent_loc_obs_comp[env_idx]
                    if current_agent_loc_obs_comp is None : 
                        current_t_step_loc_obs_embeddings_list[env_idx][agent_idx_turn, :] = torch.zeros(cfg.obs_embed_dim, device=self.device)
                        llm_response_str = "#AgentObsNone"
                        action_template_str = cfg.ACTION_TEMPLATES[-1] 
                    else:
                        # LLM grads not tracked during rollouts for obs embedding
                        loc_obs_emb = self.get_agent_observation_embedding(current_agent_loc_obs_comp) 
                        current_t_step_loc_obs_embeddings_list[env_idx][agent_idx_turn, :] = loc_obs_emb
                        
                        h_actor_current_agent_for_env = h_actor_teams[env_idx, agent_idx_turn, :].unsqueeze(0) 
                        loc_obs_emb_unsqueezed = loc_obs_emb.unsqueeze(0) 

                        # Actor and LatentNet are in eval mode and under no_grad for rollouts
                        with torch.no_grad():
                            z_roles, mu_roles, sig_roles = self.latent_net(loc_obs_emb_unsqueezed, h_actor_current_agent_for_env)
                            logits, h_actor_next_agent_for_env, _ = self.actor_net(loc_obs_emb_unsqueezed, h_actor_current_agent_for_env, z_roles)
                        
                        dist = torch.distributions.Categorical(logits=logits)
                        action_selected_idx_tensor = dist.sample()
                        action_selected_idx = action_selected_idx_tensor.item()
                        log_prob_selected = dist.log_prob(action_selected_idx_tensor).item() # log_prob of sampled action

                        current_t_step_latents_z_list[env_idx][agent_idx_turn,:,:] = z_roles.squeeze(0)
                        current_t_step_latents_mu_list[env_idx][agent_idx_turn,:,:] = mu_roles.squeeze(0)
                        current_t_step_latents_sigma_list[env_idx][agent_idx_turn,:,:] = sig_roles.squeeze(0)
                        current_t_step_actions_list[env_idx][agent_idx_turn] = action_selected_idx
                        current_t_step_log_probs_list[env_idx][agent_idx_turn] = log_prob_selected
                        h_actor_teams[env_idx, agent_idx_turn, :] = h_actor_next_agent_for_env.squeeze(0) 

                        action_template_str = cfg.ACTION_TEMPLATES[action_selected_idx]
                        logger.info(f"[Rollout][Env {env_idx}][TeamStep {t_step}][Agent {agent_idx_turn}] Selected Action: {action_template_str} (LogProb: {log_prob_selected:.3f})")
                        # generate_llm_response_for_action is already under @torch.no_grad()
                        llm_response_str = self.generate_llm_response_for_action(current_agent_loc_obs_comp, action_template_str)
                    
                    next_obs_for_next_agent, team_glob_s_after_agent_turn, team_rew, team_done, is_next_turn, info = \
                        self.env.step_agent_turn(env_idx, action_template_str, llm_response_str)

                    all_envs_current_agent_loc_obs_comp[env_idx] = next_obs_for_next_agent 
                    all_envs_current_team_global_state_comp[env_idx] = team_glob_s_after_agent_turn 
                    next_team_global_states_for_next_iteration[env_idx] = team_glob_s_after_agent_turn 

                    if not is_next_turn: 
                        self.rollout_buffer["team_rewards"][t_step, env_idx] = team_rew
                        self.rollout_buffer["team_dones"][t_step, env_idx] = team_done
                        if team_done and self.rollout_problem_llm_responses_for_csv[env_idx][0] is None:
                            task_data_csv = self.env.env_states[env_idx].get('task_data', {})
                            final_code_csv = self.env.env_states[env_idx].get("team_overall_code", "#TeamStepEnd")
                            self.rollout_problem_llm_responses_for_csv[env_idx] = (task_data_csv, final_code_csv)
                        break 

            self.rollout_buffer["local_obs_embeddings"][t_step] = torch.stack(current_t_step_loc_obs_embeddings_list)
            self.rollout_buffer["latents_z_all_roles"][t_step] = torch.stack(current_t_step_latents_z_list)
            self.rollout_buffer["latents_mu_all_roles"][t_step] = torch.stack(current_t_step_latents_mu_list)
            self.rollout_buffer["latents_sigma_all_roles"][t_step] = torch.stack(current_t_step_latents_sigma_list)
            self.rollout_buffer["actions"][t_step] = torch.stack(current_t_step_actions_list)
            self.rollout_buffer["log_probs"][t_step] = torch.stack(current_t_step_log_probs_list)
            
            h_critic_teams = h_critic_next_for_buffer # Already detached if critic_net was in no_grad or from no_grad block
            all_envs_current_team_global_state_comp = next_team_global_states_for_next_iteration

        for env_idx_csv_final in range(B): 
            if self.rollout_problem_llm_responses_for_csv[env_idx_csv_final][0] is None: 
                task_data_csv = self.env.env_states[env_idx_csv_final].get('task_data', {})
                final_code_csv = self.env.env_states[env_idx_csv_final].get("team_overall_code", "#RolloutEndNotDone")
                self.rollout_problem_llm_responses_for_csv[env_idx_csv_final] = (task_data_csv, final_code_csv)
        
        eval_data_for_csv_final: List[Tuple[Dict[str, Any], str]] = []
        for task_data_item, code_str_item in self.rollout_problem_llm_responses_for_csv:
            if task_data_item: eval_data_for_csv_final.append((task_data_item, code_str_item))

        if eval_data_for_csv_final:
            self.env.run_evaluation_and_save_csv(eval_data_for_csv_final, cfg.execution_results_csv_path)
        
        last_glob_s_T_crit_list = []
        for env_idx_gae_last in range(B):
            gs_comp = all_envs_current_team_global_state_comp[env_idx_gae_last]
            if gs_comp is not None and not self.rollout_buffer["team_dones"][cfg.num_steps_per_env -1, env_idx_gae_last]: 
                s_c_last, _ = self.get_team_global_state_embedding(gs_comp)
            else: 
                s_c_last = torch.zeros(self.cfg.global_state_dim_for_critic, device=self.device) 
            last_glob_s_T_crit_list.append(s_c_last)
        
        last_glob_s_T_crit_tensor = torch.stack(last_glob_s_T_crit_list)
        with torch.no_grad(): 
            last_team_values_for_gae, _ = self.critic_net(last_glob_s_T_crit_tensor, h_critic_teams)
            last_done_mask = self.rollout_buffer["team_dones"][cfg.num_steps_per_env -1].float() 
            last_team_values_for_gae = last_team_values_for_gae * (1.0 - last_done_mask)

        return h_actor_teams, h_critic_teams, last_team_values_for_gae


    def compute_advantages_and_returns(self, last_team_values_for_gae: torch.Tensor):
        """Computes GAE advantages and returns for team-level rewards."""
        team_advantages = torch.zeros_like(self.rollout_buffer["team_rewards"], device=self.device)
        gae_lambda, gamma = self.cfg.gae_lambda, self.cfg.gamma
        last_gae_lam_team = torch.zeros(self.cfg.num_envs, device=self.device)

        for t in reversed(range(self.cfg.num_steps_per_env)):
            if t == self.cfg.num_steps_per_env - 1:
                next_non_terminal_team = 1.0 - self.rollout_buffer["team_dones"][t].float() 
                next_team_values = last_team_values_for_gae 
            else:
                next_non_terminal_team = 1.0 - self.rollout_buffer["team_dones"][t + 1].float() 
                next_team_values = self.rollout_buffer["team_values"][t + 1] 
            
            delta_team = self.rollout_buffer["team_rewards"][t] + \
                         gamma * next_team_values * next_non_terminal_team - \
                         self.rollout_buffer["team_values"][t]
            
            team_advantages[t] = last_gae_lam_team = delta_team + \
                                   gamma * gae_lambda * next_non_terminal_team * last_gae_lam_team
        
        team_returns = team_advantages + self.rollout_buffer["team_values"]
        
        advantages_per_agent = team_advantages.unsqueeze(-1).repeat(1, 1, self.cfg.num_marl_agents)
        returns_per_agent_actor_target = team_returns.unsqueeze(-1).repeat(1, 1, self.cfg.num_marl_agents)
        
        return advantages_per_agent, returns_per_agent_actor_target, team_returns

    def ppo_update(self, advantages_agent_flat: torch.Tensor, returns_agent_actor_target_flat: torch.Tensor,
                   local_obs_flat: torch.Tensor,
                   global_states_critic_flat_team: torch.Tensor, global_states_inference_flat_team: torch.Tensor,
                   actor_h_flat: torch.Tensor, critic_h_team_flat: torch.Tensor,
                   actions_agent_flat: torch.Tensor, log_probs_old_agent_flat: torch.Tensor,
                   team_returns_flat_target_critic_inf: torch.Tensor, team_values_old_flat_for_clip: torch.Tensor):
        """Performs PPO updates for Actor, Critic, LatentNet, and InferenceNet, referencing SHPPO Algorithm 2."""
        cfg, device = self.cfg, self.device
        num_agent_samples_total = local_obs_flat.shape[0]
        num_team_samples_total = global_states_critic_flat_team.shape[0]

        if cfg.norm_adv:
            advantages_agent_flat = (advantages_agent_flat - advantages_agent_flat.mean()) / (advantages_agent_flat.std() + 1e-9)

        agent_data_minibatch_size = num_agent_samples_total // cfg.num_minibatches
        team_data_minibatch_size = num_team_samples_total // cfg.num_minibatches

        if agent_data_minibatch_size == 0 or team_data_minibatch_size == 0:
            logger.warning("Minibatch size is 0, skipping PPO update.")
            return

        for _ in range(cfg.ppo_epochs):
            perm_agent_indices = torch.randperm(num_agent_samples_total, device=device)
            perm_team_indices = torch.randperm(num_team_samples_total, device=device)

            # --- Actor and LLM LoRA Update (Loop over agent data minibatches) ---
            for start_idx in range(0, num_agent_samples_total, agent_data_minibatch_size):
                end_idx = min(start_idx + agent_data_minibatch_size, num_agent_samples_total)
                mb_agent_indices = perm_agent_indices[start_idx:end_idx]
                if mb_agent_indices.numel() == 0: continue

                # Detach inputs for this minibatch to prevent graph interference
                obs_mb_loc_actor = local_obs_flat[mb_agent_indices].detach()
                h_actor_mb_actor = actor_h_flat[mb_agent_indices].detach()
                
                actions_mb_agent = actions_agent_flat[mb_agent_indices]
                old_log_probs_mb_agent = log_probs_old_agent_flat[mb_agent_indices]
                adv_mb_agent = advantages_agent_flat[mb_agent_indices]

                # For ActorNet, latents z_roles_for_policy are obtained via LatentNet in no_grad mode.
                # This means ActorNet does not train LatentNet parameters directly through its own loss.
                with torch.no_grad():
                    z_roles_for_policy, _, _ = self.latent_net(obs_mb_loc_actor, h_actor_mb_actor)
                
                # ActorNet forward pass. If LLM LoRA is trained, obs_mb_loc_actor (via prompt_emb) will carry grads.
                current_logits, _, _ = self.actor_net(obs_mb_loc_actor, h_actor_mb_actor, z_roles_for_policy)
                
                current_dist = torch.distributions.Categorical(logits=current_logits)
                new_log_probs_agent = current_dist.log_prob(actions_mb_agent)
                entropy_bonus_actor = current_dist.entropy().mean()

                ratio = torch.exp(new_log_probs_agent - old_log_probs_mb_agent)
                surr1 = ratio * adv_mb_agent
                surr2 = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv_mb_agent
                policy_loss_agent = -torch.min(surr1, surr2).mean()
                actor_loss_total = policy_loss_agent - cfg.ent_coef * entropy_bonus_actor

                self.opt_actor.zero_grad()
                if self.opt_llm_lora: self.opt_llm_lora.zero_grad()
                
                actor_loss_total.backward() 

                actor_params_to_clip = list(self.actor_net.parameters())
                if not isinstance(self.state_projection_layer, nn.Identity):
                    actor_params_to_clip += list(self.state_projection_layer.parameters())
                if cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(actor_params_to_clip, cfg.max_grad_norm)

                if self.opt_llm_lora:
                    llm_lora_grad_params = [p for p in self.llm_model.parameters() if p.requires_grad and p.grad is not None]
                    if llm_lora_grad_params and cfg.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(llm_lora_grad_params, cfg.max_grad_norm)
                    self.opt_llm_lora.step()
                self.opt_actor.step()

            # --- Critic, LatentNet, InferenceNet Updates (Loop over team data minibatches) ---
            # This loop aligns with Algorithm 2's "Sample a random minibatch..." and subsequent updates.
            for start_idx in range(0, num_team_samples_total, team_data_minibatch_size):
                end_idx = min(start_idx + team_data_minibatch_size, num_team_samples_total)
                mb_team_indices = perm_team_indices[start_idx:end_idx]
                if mb_team_indices.numel() == 0: continue

                # Detach inputs for this minibatch
                gs_critic_mb = global_states_critic_flat_team[mb_team_indices].detach()
                h_critic_mb_team = critic_h_team_flat[mb_team_indices].detach()
                returns_mb_team_target = team_returns_flat_target_critic_inf[mb_team_indices]
                values_old_team_mb_for_clip = team_values_old_flat_for_clip[mb_team_indices]
                
                agent_indices_for_lat_inf_list = []
                for team_idx_val in mb_team_indices.tolist():
                    for agent_i in range(cfg.num_marl_agents):
                        agent_indices_for_lat_inf_list.append(team_idx_val * cfg.num_marl_agents + agent_i)
                
                mb_agent_indices_for_lat_inf = torch.tensor(agent_indices_for_lat_inf_list, device=device, dtype=torch.long)
                if mb_agent_indices_for_lat_inf.numel() == 0: continue

                obs_mb_loc_for_lat_inf = local_obs_flat[mb_agent_indices_for_lat_inf].detach()
                h_actor_mb_for_lat_inf = actor_h_flat[mb_agent_indices_for_lat_inf].detach()
                gs_inference_mb = global_states_inference_flat_team[mb_team_indices].detach()

                # --- 2) CriticNet Update (Equation 14) ---
                current_team_values, _ = self.critic_net(gs_critic_mb, h_critic_mb_team)
                values_pred_squeezed = current_team_values.squeeze() if current_team_values.ndim > 1 else current_team_values
                if cfg.clip_vloss:
                    values_pred_clipped = values_old_team_mb_for_clip + torch.clamp(
                        values_pred_squeezed - values_old_team_mb_for_clip, -cfg.clip_coef, cfg.clip_coef)
                    vf_loss_unclipped = F.mse_loss(values_pred_squeezed, returns_mb_team_target)
                    vf_loss_clipped = F.mse_loss(values_pred_clipped, returns_mb_team_target)
                    critic_loss = torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
                else:
                    critic_loss = F.mse_loss(values_pred_squeezed, returns_mb_team_target).mean()
                critic_loss_final = critic_loss * cfg.vf_coef
                
                self.opt_critic.zero_grad()
                critic_loss_final.backward()
                if cfg.max_grad_norm > 0: nn.utils.clip_grad_norm_(self.critic_net.parameters(), cfg.max_grad_norm)
                self.opt_critic.step()

                # --- LatentNet & InferenceNet Updates (Equations 6, 7, 8, 10, 11) ---
                # LatentNet forward pass (for Eq 7, 8)
                z_latents, mu_latents, sigma_latents = self.latent_net(obs_mb_loc_for_lat_inf, h_actor_mb_for_lat_inf)
                latent_entropy = (0.5 * cfg.latent_dim * (1 + math.log(2 * math.pi)) + torch.log(sigma_latents + 1e-9).sum(dim=-1)).mean() # L_e
                latent_diversity = cosine_diversity(z_latents) # L_d
                
                current_team_minibatch_size = mb_team_indices.size(0)
                mu_latents_team_view = mu_latents.view(current_team_minibatch_size, cfg.num_marl_agents, cfg.N_ACTION_TEMPLATES, cfg.latent_dim)
                sigma_latents_team_view = sigma_latents.view(current_team_minibatch_size, cfg.num_marl_agents, cfg.N_ACTION_TEMPLATES, cfg.latent_dim)
                
                # Compute L_v (Eq 6) for LatentNet loss (Eq 10)
                # Gradients will flow from V_I_for_latent_loss to mu_latents_team_view & sigma_latents_team_view,
                # and thus to LatentNet parameters. InferenceNet parameters are also part of this graph.
                V_I_for_latent_loss = self.inference_net(gs_inference_mb, mu_latents_team_view, sigma_latents_team_view)
                
                loss_L_v = -V_I_for_latent_loss.mean() * cfg.lambda_V_I_guidance_for_latent # This is -L_v term in Eq 10
                loss_L_e = cfg.lambda_e_latent * latent_entropy
                loss_L_d = -cfg.lambda_d_latent * latent_diversity
                loss_latent_net = loss_L_v + loss_L_e + loss_L_d
                
                self.opt_latent.zero_grad()
                # Important: Grads for InferenceNet might be populated by loss_latent_net.backward()
                # These are zeroed out before InferenceNet's own update.
                loss_latent_net.backward(retain_graph=True) # Retain graph for InferenceNet's own update
                if cfg.max_grad_norm > 0: nn.utils.clip_grad_norm_(self.latent_net.parameters(), cfg.max_grad_norm)
                self.opt_latent.step() 

                # Compute L_I (Eq 11) for InferenceNet update
                # Use detached latents so InferenceNet loss doesn't backprop to LatentNet parameters
                V_I_for_inference_loss = self.inference_net(gs_inference_mb, 
                                                            mu_latents_team_view.detach(), 
                                                            sigma_latents_team_view.detach())
                loss_inference_net = F.mse_loss(V_I_for_inference_loss.squeeze(), returns_mb_team_target) * cfg.lambda_inf_mse
                
                self.opt_inference.zero_grad() 
                loss_inference_net.backward() 
                if cfg.max_grad_norm > 0: nn.utils.clip_grad_norm_(self.inference_net.parameters(), cfg.max_grad_norm)
                self.opt_inference.step()


    def train(self, total_timesteps_override: Optional[int] = None):
        """Main training loop."""
        cfg=self.cfg
        total_training_team_steps = total_timesteps_override if total_timesteps_override is not None else cfg.total_timesteps
        
        if cfg.wandb_project_name:
            try:
                wandb.init(project=cfg.wandb_project_name, name=cfg.wandb_run_name, config=vars(cfg), reinit=True)
            except Exception as e:
                logger.error(f"WandB initialization failed: {e}. Disabling WandB.")
                cfg.wandb_project_name = None 
        
        next_h_actor_teams = torch.zeros(cfg.num_envs, cfg.num_marl_agents, cfg.actor_rnn_hidden_dim, device=self.device)
        next_h_critic_teams = torch.zeros(cfg.num_envs, cfg.critic_rnn_hidden_dim, device=self.device)
        
        num_team_steps_per_update = cfg.num_envs * cfg.num_steps_per_env
        if num_team_steps_per_update == 0 :
            logger.error("num_envs * num_steps_per_env is 0. Cannot collect rollouts for training.")
            return
            
        num_total_updates = total_training_team_steps // num_team_steps_per_update

        if num_total_updates == 0:
            logger.warning(f"Not enough total_timesteps ({total_training_team_steps}) for a single PPO update cycle ({num_team_steps_per_update} steps needed). Training will not start.")
            return
            
        logger.info(f"Starting training for {num_total_updates} PPO update cycles ({total_training_team_steps} total team environment steps).")
        
        for update_cycle_idx in range(1, num_total_updates + 1):
            current_total_env_steps = update_cycle_idx * num_team_steps_per_update
            
            if self.opt_llm_lora: self.llm_model.eval()
            self.actor_net.eval(); self.critic_net.eval(); self.latent_net.eval(); self.inference_net.eval()
            if not isinstance(self.state_projection_layer, nn.Identity): self.state_projection_layer.eval()

            h_actor_end_rollout, h_critic_end_rollout, last_team_values_gae = self.collect_rollouts(next_h_actor_teams, next_h_critic_teams)
            
            adv_agent, ret_agent_actor_tgt, team_ret_crit_inf_tgt = self.compute_advantages_and_returns(last_team_values_gae)
            
            self.actor_net.train(); self.critic_net.train(); self.latent_net.train(); self.inference_net.train()
            if self.opt_llm_lora: self.llm_model.train()
            if not isinstance(self.state_projection_layer, nn.Identity): self.state_projection_layer.train()
            
            total_agent_samples_in_rollout = cfg.num_steps_per_env * cfg.num_envs * cfg.num_marl_agents
            total_team_samples_in_rollout = cfg.num_steps_per_env * cfg.num_envs
            
            flat_local_obs = self.rollout_buffer["local_obs_embeddings"].reshape(total_agent_samples_in_rollout, cfg.obs_embed_dim)
            flat_actor_h = self.rollout_buffer["actor_hidden_states"].reshape(total_agent_samples_in_rollout, cfg.actor_rnn_hidden_dim)
            flat_actions_agent = self.rollout_buffer["actions"].reshape(total_agent_samples_in_rollout)
            flat_log_probs_old_agent = self.rollout_buffer["log_probs"].reshape(total_agent_samples_in_rollout)
            flat_adv_agent = adv_agent.reshape(total_agent_samples_in_rollout)
            flat_ret_agent_actor_tgt = ret_agent_actor_tgt.reshape(total_agent_samples_in_rollout)
            
            flat_global_states_critic_team = self.rollout_buffer["global_state_embeddings_critic"].reshape(total_team_samples_in_rollout, cfg.global_state_dim_for_critic)
            flat_global_states_inference_team = self.rollout_buffer["global_state_embeddings_inference"].reshape(total_team_samples_in_rollout, cfg.global_state_dim_for_inference)
            flat_critic_h_team = self.rollout_buffer["critic_hidden_states"].reshape(total_team_samples_in_rollout, cfg.critic_rnn_hidden_dim)
            flat_team_returns_target_critic_inf = team_ret_crit_inf_tgt.reshape(total_team_samples_in_rollout)
            flat_team_values_old_for_clip = self.rollout_buffer["team_values"].reshape(total_team_samples_in_rollout).detach().clone()
            
            self.ppo_update(
                flat_adv_agent, flat_ret_agent_actor_tgt, flat_local_obs, 
                flat_global_states_critic_team, flat_global_states_inference_team,
                flat_actor_h, flat_critic_h_team,
                flat_actions_agent, flat_log_probs_old_agent,
                flat_team_returns_target_critic_inf, flat_team_values_old_for_clip
            )
            
            next_h_actor_teams = h_actor_end_rollout.detach()
            next_h_critic_teams = h_critic_end_rollout.detach()
            
            avg_rollout_team_reward = self.rollout_buffer["team_rewards"].mean().item()
            if cfg.wandb_project_name and wandb.run:
                wandb.log({
                    "rollout/avg_team_reward": avg_rollout_team_reward,
                    "global_step": current_total_env_steps,
                    "update_cycle": update_cycle_idx
                })
                
            if update_cycle_idx % 10 == 0:
                logger.info(f"[Update {update_cycle_idx}/{num_total_updates}, EnvTeamSteps {current_total_env_steps}] Avg Team Rollout Reward: {avg_rollout_team_reward:.3f}")
            
            if update_cycle_idx > 0 and update_cycle_idx % cfg.evaluate_interval == 0:
                self.evaluate_model(current_total_env_steps)
            
            if update_cycle_idx > 0 and update_cycle_idx % (cfg.evaluate_interval * 10) == 0: # More frequent checkpoint saving
                self.save_models(f"shppo_checkpoint_update_{update_cycle_idx}")
                
        logger.info("Training complete.")
        self.save_models("shppo_final_model")
        if cfg.wandb_project_name and wandb.run:
            wandb.finish()

    def evaluate_model(self, current_global_step: int, eval_csv_filepath: Optional[str] = None):
        """Evaluates the current model deterministically on a subset of tasks."""
        cfg = self.cfg
        log_path = eval_csv_filepath if eval_csv_filepath else f"eval_results_step_{current_global_step}.csv"
        logger.info(f"\n--- Evaluation at Global Step {current_global_step}, saving results to {log_path} ---")
        
        self.actor_net.eval(); self.latent_net.eval(); self.critic_net.eval(); self.inference_net.eval()
        if self.opt_llm_lora: self.llm_model.eval()
        if not isinstance(self.state_projection_layer, nn.Identity): self.state_projection_layer.eval()
        
        num_eval_problems_to_run = min(cfg.evaluate_episodes, len(self.env.all_problem_tasks))
        if num_eval_problems_to_run == 0:
            logger.info("No problems available or configured for evaluation.")
            return
            
        eval_problem_indices = random.sample(range(len(self.env.all_problem_tasks)), num_eval_problems_to_run)
        
        all_episode_team_rewards: List[float] = []
        all_episode_team_pass_fractions: List[float] = []
        problem_task_code_pairs_for_csv: List[Tuple[Dict[str, Any], str]] = []
        
        eval_env_idx = 0 

        for i in range(num_eval_problems_to_run):
            problem_task_for_eval = self.env.all_problem_tasks[eval_problem_indices[i]]
            logger.info(f"Evaluating task: {problem_task_for_eval.get('name', 'Unknown Task')}")

            current_agent_loc_obs_comp_eval, current_team_global_state_comp_eval = \
                self.env._reset_one_env(eval_env_idx, problem_task_for_eval)

            h_actor_eval_env = torch.zeros(1, cfg.num_marl_agents, cfg.actor_rnn_hidden_dim, device=self.device)

            episode_team_reward = 0.0
            final_code_for_this_eval_episode = ""

            for _ in range(cfg.max_team_episode_steps): 
                if self.env.env_states[eval_env_idx].get("episode_done", False):
                    break 

                for agent_idx_turn_eval in range(cfg.num_marl_agents):
                    if current_agent_loc_obs_comp_eval is None: 
                        action_str_eval = cfg.ACTION_TEMPLATES[-1] 
                        llm_response_eval = "#EvalAgentObsNone"
                    else:
                        loc_obs_emb_eval = self.get_agent_observation_embedding(current_agent_loc_obs_comp_eval).unsqueeze(0) 
                        h_actor_this_agent_eval = h_actor_eval_env[0, agent_idx_turn_eval, :].unsqueeze(0) 
                        
                        with torch.no_grad():
                            z_roles_eval, _, _ = self.latent_net(loc_obs_emb_eval, h_actor_this_agent_eval)
                            logits_eval, h_actor_next_unsqueeze_eval, _ = self.actor_net(loc_obs_emb_eval, h_actor_this_agent_eval, z_roles_eval)
                        
                        action_idx_eval = torch.argmax(logits_eval, dim=1).item()
                        action_str_eval = cfg.ACTION_TEMPLATES[action_idx_eval]
                        
                        h_actor_eval_env[0, agent_idx_turn_eval, :] = h_actor_next_unsqueeze_eval.squeeze(0).detach()
                        logger.info(f"[Evaluate][Task: {problem_task_for_eval.get('name', f'EvalTask{i}')}][Agent {agent_idx_turn_eval}] Selected Action: {action_str_eval}")
                        llm_response_eval = self.generate_llm_response_for_action(
                            agent_local_obs_components=current_agent_loc_obs_comp_eval,
                            action_template=action_str_eval
                        )
                    
                    next_obs_for_next_agent_eval, team_glob_s_after_agent_turn_eval, \
                    team_reward_this_turn_eval, team_done_this_episode_eval, \
                    is_next_turn_in_team_eval, info_eval = \
                        self.env.step_agent_turn(eval_env_idx, action_str_eval, llm_response_eval)
                    
                    current_agent_loc_obs_comp_eval = next_obs_for_next_agent_eval 
                    current_team_global_state_comp_eval = team_glob_s_after_agent_turn_eval

                    if not is_next_turn_in_team_eval: 
                        episode_team_reward += team_reward_this_turn_eval
                        final_code_for_this_eval_episode = self.env.env_states[eval_env_idx].get("team_overall_code", "")
                        if team_done_this_episode_eval:
                            break 
                
                if self.env.env_states[eval_env_idx].get("episode_done", False): 
                    break
            
            all_episode_team_rewards.append(episode_team_reward)
            all_episode_team_pass_fractions.append(self.env.env_states[eval_env_idx].get("team_current_pass_fraction", 0.0))
            problem_task_code_pairs_for_csv.append((problem_task_for_eval, final_code_for_this_eval_episode))
        
        if problem_task_code_pairs_for_csv:
            self.env.run_evaluation_and_save_csv(problem_task_code_pairs_for_csv, log_path)
            
        avg_team_pass_fraction_eval = np.mean(all_episode_team_pass_fractions) if all_episode_team_pass_fractions else 0.0
        avg_team_reward_eval = np.mean(all_episode_team_rewards) if all_episode_team_rewards else 0.0
        
        logger.info(f"Evaluation Summary: Avg Team Pass Fraction = {avg_team_pass_fraction_eval:.3f}, Avg Team Reward = {avg_team_reward_eval:.3f}")
        if cfg.wandb_project_name and wandb.run:
            wandb.log({
                "eval/avg_team_pass_fraction": avg_team_pass_fraction_eval,
                "eval/avg_team_reward": avg_team_reward_eval,
                "global_step": current_global_step
            })
        logger.info("--- End Evaluation ---")

    def save_models(self, path_prefix: str ="shppo_model"):
        """Saves all network models and LoRA adapters."""
        os.makedirs(path_prefix, exist_ok=True)
        torch.save(self.actor_net.state_dict(), os.path.join(path_prefix, "actor_net.pth"))
        torch.save(self.critic_net.state_dict(), os.path.join(path_prefix, "critic_net.pth"))
        torch.save(self.latent_net.state_dict(), os.path.join(path_prefix, "latent_net.pth"))
        torch.save(self.inference_net.state_dict(), os.path.join(path_prefix, "inference_net.pth"))
        
        if not isinstance(self.state_projection_layer, nn.Identity):
            torch.save(self.state_projection_layer.state_dict(), os.path.join(path_prefix, "state_projection_layer.pth"))
            
        if self.opt_llm_lora and hasattr(self.llm_model, "save_pretrained"):
            try:
                llm_lora_path = os.path.join(path_prefix, "llm_lora_adapters")
                self.llm_model.save_pretrained(llm_lora_path)
                self.tokenizer.save_pretrained(llm_lora_path) 
                logger.info(f"LLM LoRA adapters and tokenizer saved to {llm_lora_path}")
            except Exception as e:
                logger.error(f"LLM LoRA adapter saving failed: {e}")
        logger.info(f"All models saved with prefix {path_prefix}")

    def load_models(self, path_prefix: str ="shppo_model", load_llm_adapters: bool =True):
        """Loads all network models and optionally LoRA adapters."""
        try:
            self.actor_net.load_state_dict(torch.load(os.path.join(path_prefix, "actor_net.pth"), map_location=self.device))
            self.critic_net.load_state_dict(torch.load(os.path.join(path_prefix, "critic_net.pth"), map_location=self.device))
            self.latent_net.load_state_dict(torch.load(os.path.join(path_prefix, "latent_net.pth"), map_location=self.device))
            self.inference_net.load_state_dict(torch.load(os.path.join(path_prefix, "inference_net.pth"), map_location=self.device))
            
            state_proj_path = os.path.join(path_prefix, "state_projection_layer.pth")
            if not isinstance(self.state_projection_layer, nn.Identity) and os.path.exists(state_proj_path):
                self.state_projection_layer.load_state_dict(torch.load(state_proj_path, map_location=self.device))
            logger.info(f"Policy, Value, Latent, and Inference networks loaded from {path_prefix}")

            if load_llm_adapters and self.opt_llm_lora: 
                llm_lora_adapter_path = os.path.join(path_prefix, "llm_lora_adapters")
                if os.path.exists(llm_lora_adapter_path):
                    base_model = self.llm_model.base_model.model if hasattr(self.llm_model, "base_model") else self.llm_model
                    self.llm_model = PeftModel.from_pretrained(base_model, llm_lora_adapter_path, is_trainable=True) # type: ignore [no-untyped-call]
                    logger.info(f"LLM LoRA adapters loaded from {llm_lora_adapter_path}")
                else:
                    logger.warning(f"LLM LoRA adapters not found at {llm_lora_adapter_path}. Skipping LoRA adapter loading.")
        except FileNotFoundError as e:
            logger.error(f"Failed to load models: {e}. Ensure paths are correct and all model files exist at prefix {path_prefix}.")
        except Exception as e:
            logger.error(f"An error occurred during model loading from {path_prefix}: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    torch.backends.cudnn.benchmark = True # type: ignore [misc] 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    config = SHPPOConfig()
    # Example override for quick testing:
    config.num_marl_agents = 3
    config.total_timesteps = 2000 
    config.num_envs = 4
    config.num_steps_per_env = 2 
    config.ppo_epochs = 2
    config.num_minibatches = 2 
    
    # ---- ADDED/MODIFIED Configs for Global State ----
    config.max_prompt_length_for_global_embedding = getattr(config, 'max_prompt_length_for_global_embedding', 256) # Max length for prompt in global state
    config.num_scalar_global_features = getattr(config, 'num_scalar_global_features', 2) # For pass_fraction, episode_progress
    # global_state_dim_for_critic and global_state_dim_for_inference will be set in build_networks
    # after llm_actual_hidden_size is known.
    # ---- END Config Additions ----

    config.__post_init__() 

    config.dataset_max_problems = config.num_envs * 2 
    config.evaluate_episodes = max(1, config.num_envs) 
    config.evaluate_interval = 5 
    config.wandb_project_name = None 
    


    try:
        dataset_loader = CodeContestDataset(
            split="train", 
            max_problems=config.dataset_max_problems, 
            max_cases=config.dataset_max_cases, 
            cache_dir=None 
        )
        all_problem_tasks_for_env = dataset_loader.get_all_tasks()
        if not all_problem_tasks_for_env: 
            raise ValueError("No tasks loaded from dataset. Check dataset_max_problems or dataset source.")
        if len(all_problem_tasks_for_env) < config.num_envs:
            logger.warning(
                f"Number of loaded unique tasks ({len(all_problem_tasks_for_env)}) is less than num_envs ({config.num_envs}). "
                "Problems will be sampled with replacement during environment resets."
            )
    except Exception as e: 
        logger.error(f"Dataset loading failed: {e}", exc_info=True)
        sys.exit(1)
    
    shppo_env_instance = SHPPOCodeEnv(config=config, all_problem_tasks=all_problem_tasks_for_env)
    trainer = SHPPOTrainer(cfg=config, device=device, env=shppo_env_instance) # cfg is updated in build_networks called by __init__
    
    try:
        logger.info("Starting SHPPO training...")
        trainer.train(total_timesteps_override=config.total_timesteps)
        logger.info("Training finished successfully.")
        trainer.save_models("shppo_final_model_after_train")
        logger.info("Final models saved to 'shppo_final_model_after_train'.")
    except KeyboardInterrupt: 
        logger.info("Training interrupted by user (KeyboardInterrupt). Saving current models...")
        trainer.save_models("shppo_model_interrupted")
        logger.info("Models saved to 'shppo_model_interrupted'.")
    except Exception as e: 
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        trainer.save_models("shppo_model_error")
        logger.info("Models saved to 'shppo_model_error' due to error.")