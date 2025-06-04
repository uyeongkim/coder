import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional, Union
from torch.distributions import Categorical, Normal
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from marcog_config import (
    BaseRoleConfig, PlannerConfig, CoderConfig, DebuggerConfig, GlobalSHPPOConfig,
    ACTION_TO_IDX, IDX_TO_ACTION, ACTION_METADATA,
    PLANNER_ACTION_INDICES, CODER_ACTION_INDICES, DEBUGGER_ACTION_INDICES, ACTION_TEMPLATES
)
from marcog_test import CodeContestDataset, CodeTester 

import random
import pandas as pd
import numpy as np
import wandb
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

def ortho_init(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        layers_list = []
        current_dim = input_dim
        if num_layers == 1:
            layers_list.append(nn.Linear(current_dim, output_dim))
        elif num_layers > 1:
            for _ in range(num_layers - 1):
                layers_list.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
                current_dim = hidden_dim
            layers_list.append(nn.Linear(current_dim, output_dim))
        else:
            raise ValueError("num_layers in MLPBlock must be >= 1")
        self.net = nn.Sequential(*layers_list)
        self.net.apply(lambda m: ortho_init(m, math.sqrt(2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.net, '0') and isinstance(self.net[0], nn.Linear):
            expected_dtype = self.net[0].weight.dtype
            return self.net(x.to(expected_dtype))
        return self.net(x)

# --- SHPPO Network Modules ---
class Encoder(nn.Module):
    def __init__(self, role_config: BaseRoleConfig, global_config: GlobalSHPPOConfig):
        super().__init__()
        input_dim = role_config.obs_embed_dim + global_config.actor_rnn_hidden_dim
        self.encoder_mlp = MLPBlock(input_dim, global_config.mlp_hidden_dim,
                                    hidden_dim=global_config.mlp_hidden_dim, num_layers=3)
        self.fc_mu = nn.Linear(global_config.mlp_hidden_dim,
                               role_config.N_ACTION_TEMPLATES * global_config.latent_dim)
        self.fc_sigma = nn.Linear(global_config.mlp_hidden_dim,
                                  role_config.N_ACTION_TEMPLATES * global_config.latent_dim)
        self.fc_mu.apply(lambda m: ortho_init(m, 0.01))
        self.fc_sigma.apply(lambda m: ortho_init(m, 0.01))

    def forward(self, obs_emb: torch.Tensor, h_actor_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dtype = obs_emb.dtype
        if obs_emb.dim() == 3:                     
            B, T, F = obs_emb.shape
            obs_emb = obs_emb.view(B * T, F)        

            if h_actor_prev.dim() == 2:             
                h_actor_prev = h_actor_prev.unsqueeze(1) \
                                       .repeat(1, T, 1)      
            if h_actor_prev.dim() == 3:             
                h_actor_prev = h_actor_prev.view(B * T, -1) 

        elif h_actor_prev.dim() == 3:
            h_actor_prev = h_actor_prev.squeeze(1)           

        x = torch.cat([obs_emb, h_actor_prev.to(target_dtype)], dim=-1)  
        x = self.encoder_mlp(x)
        mu_flat    = self.fc_mu(x)
        sigma_flat = self.fc_sigma(x)
        return mu_flat, sigma_flat

class LatentNet(nn.Module):
    def __init__(self, encoder: Encoder, role_config: BaseRoleConfig, global_config: GlobalSHPPOConfig):
        super().__init__()
        self.encoder = encoder
        self.N_ACTION_TEMPLATES = role_config.N_ACTION_TEMPLATES
        self.latent_dim = global_config.latent_dim
        self.model_dtype = global_config.model_dtype

    def forward(self, obs_emb: torch.Tensor, h_actor_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_dtype = self.model_dtype if self.model_dtype is not None else obs_emb.dtype
        
        mu_flat, sigma_raw_flat = self.encoder(obs_emb.to(target_dtype), h_actor_prev.to(target_dtype))
        current_batch_size = obs_emb.size(0)
        
        mu = mu_flat.view(current_batch_size, self.N_ACTION_TEMPLATES, self.latent_dim)
        sigma_raw = sigma_raw_flat.view(current_batch_size, self.N_ACTION_TEMPLATES, self.latent_dim)
        
        sigma = F.softplus(sigma_raw) + 1e-5 
        epsilon = torch.randn_like(sigma, dtype=target_dtype)
        z = mu + sigma * epsilon 
        return z, mu, sigma

class InferenceNet(nn.Module):
    def __init__(self, global_config: GlobalSHPPOConfig):
        super().__init__()
        self.global_config = global_config
        total_latent_params_dim_from_all_agents = 0
        if global_config.planner_role_config and global_config.total_planner_agents > 0:
            total_latent_params_dim_from_all_agents += (global_config.total_planner_agents *
                                        global_config.planner_role_config.N_ACTION_TEMPLATES *
                                        global_config.latent_dim)
        if global_config.coder_role_config and global_config.total_coder_agents > 0:
            total_latent_params_dim_from_all_agents += (global_config.total_coder_agents *
                                        global_config.coder_role_config.N_ACTION_TEMPLATES *
                                        global_config.latent_dim)
        if global_config.debugger_role_config and global_config.total_debugger_agents > 0:
            total_latent_params_dim_from_all_agents += (global_config.total_debugger_agents *
                                        global_config.debugger_role_config.N_ACTION_TEMPLATES *
                                        global_config.latent_dim)
        input_dim = global_config.global_state_dim + (total_latent_params_dim_from_all_agents * 2)
        logger.info(f"InferenceNet input_dim calculated: {input_dim}")
        self.v_head = MLPBlock(input_dim, 1, hidden_dim=global_config.mlp_hidden_dim, num_layers=3)
        self.v_head.apply(lambda m: ortho_init(m, 1.0))

    def forward(self,
                  glob_s_emb: torch.Tensor,
                  concatenated_all_agents_mu_flat: torch.Tensor,
                  concatenated_all_agents_sigma_flat: torch.Tensor
                 ) -> torch.Tensor:
        target_dtype = glob_s_emb.dtype
        x = torch.cat([glob_s_emb, 
                       concatenated_all_agents_mu_flat.to(target_dtype), 
                       concatenated_all_agents_sigma_flat.to(target_dtype)], dim=-1)
        return self.v_head(x).squeeze(-1)

class HeteLayerDecoder(nn.Module):
    def __init__(self, global_config: GlobalSHPPOConfig):
        super().__init__()
        self.global_config = global_config
        self.w_decoder = nn.Linear(global_config.latent_dim,
                                   global_config.hete_layer_input_dim * global_config.hete_layer_output_dim)
        self.b_decoder = nn.Linear(global_config.latent_dim, global_config.hete_layer_output_dim)
        self.w_decoder.apply(lambda m: ortho_init(m, math.sqrt(0.1)))
        self.b_decoder.apply(lambda m: ortho_init(m, math.sqrt(0.1)))

    def forward(self, z_template_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w_decoder_dtype = self.w_decoder.weight.dtype
        weights_raw = self.w_decoder(z_template_flat.to(w_decoder_dtype))
        biases_raw = self.b_decoder(z_template_flat.to(w_decoder_dtype))
        
        weights = weights_raw.view(-1, 
                                   self.global_config.hete_layer_output_dim,
                                   self.global_config.hete_layer_input_dim)
        biases = biases_raw
        return weights, biases

class ActorNet(nn.Module):
    def __init__(self, role_config: BaseRoleConfig, global_config: GlobalSHPPOConfig):
        super().__init__()
        self.role_config   = role_config
        self.global_config = global_config
        self.model_dtype   = global_config.model_dtype  


        self.obs_encoder = MLPBlock(
            role_config.obs_embed_dim,
            global_config.actor_rnn_hidden_dim,
            hidden_dim = global_config.mlp_hidden_dim,
            num_layers = 1,
        )
        self.rnn = nn.GRU(
            global_config.actor_rnn_hidden_dim,
            global_config.actor_rnn_hidden_dim,
            batch_first = True,
        )
        self.hete_layer_decoder = HeteLayerDecoder(global_config)
        self.final_mlp = MLPBlock(
            global_config.hete_layer_output_dim,
            global_config.actor_final_mlp_output_dim,
            hidden_dim = global_config.mlp_hidden_dim,
            num_layers = 1,
        )
        self.policy_head = nn.Linear(
            global_config.actor_final_mlp_output_dim, 1
        )
        self.policy_head.apply(lambda m: ortho_init(m, 0.01))


    def forward(
        self,
        agent_obs_emb: torch.Tensor,                 
        agent_h_prev : torch.Tensor,                 
        z_all_templates_for_this_agent: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        
        dtype = self.model_dtype or agent_obs_emb.dtype
        B, N_tpl, D_z = z_all_templates_for_this_agent.shape

        
        if agent_h_prev.dim() == 2:            
            agent_h_prev = agent_h_prev.unsqueeze(0)
        elif agent_h_prev.dim() == 3 and agent_h_prev.size(1) == 1:
            agent_h_prev = agent_h_prev.permute(1, 0, 2)  

        
        obs_feat = self.obs_encoder(agent_obs_emb.to(dtype))          
        rnn_out , h_next = self.rnn(obs_feat.unsqueeze(1), agent_h_prev.to(dtype))
        h_t = rnn_out.squeeze(1)                                      

        h_rep = h_t.unsqueeze(1).expand(B, N_tpl, -1)\
                                 .reshape(-1, self.global_config.hete_layer_input_dim) 
        z_flat = z_all_templates_for_this_agent.reshape(-1, D_z).to(dtype)              

        W, b = self.hete_layer_decoder(z_flat)  
        hete = torch.bmm(h_rep.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b          
        feat_flat = self.final_mlp(hete)                                               
        feat_by_tpl = feat_flat.view(B, N_tpl, -1)                                     

        role_indices = (
            PLANNER_ACTION_INDICES  if self.role_config.role_name == "planner" else
            CODER_ACTION_INDICES    if self.role_config.role_name == "coder"   else
            DEBUGGER_ACTION_INDICES
        )
        valid_tpl_ids = sorted(role_indices)

        logits_chunks = []
        for local_idx, global_idx in enumerate(valid_tpl_ids):
            tpl_feat = feat_by_tpl[:, local_idx, :]
            n_sample = ACTION_METADATA[IDX_TO_ACTION[global_idx]]["sample_count"]
            tpl_feat_exp = tpl_feat.unsqueeze(1).expand(-1, n_sample, -1)\
                                 .reshape(-1, self.global_config.actor_final_mlp_output_dim)

            logits_flat = self.policy_head(tpl_feat_exp.to(self.policy_head.weight.dtype))
            logits_chunks.append(logits_flat.view(B, n_sample))

        action_logits = torch.cat(logits_chunks, dim=1) if logits_chunks else \
                        torch.empty(B, 0, device=agent_obs_emb.device, dtype=dtype)

        action_probs = F.softmax(action_logits.to(torch.float32), dim=-1)

        return action_logits, h_next.squeeze(0).to(dtype), action_probs

class CriticNet(nn.Module):
    def __init__(self, global_config: GlobalSHPPOConfig):
        super().__init__()
        self.global_config = global_config
        self.model_dtype = global_config.model_dtype
        self.global_state_projector = MLPBlock(global_config.global_state_dim, global_config.critic_rnn_hidden_dim,
                                               hidden_dim=global_config.mlp_hidden_dim, num_layers=2)
        self.rnn = nn.GRU(global_config.critic_rnn_hidden_dim, global_config.critic_rnn_hidden_dim, batch_first=True)
        self.value_head = MLPBlock(global_config.critic_rnn_hidden_dim, 1,
                                   hidden_dim=global_config.mlp_hidden_dim, num_layers=1)
        self.value_head.apply(lambda m: ortho_init(m, 1.0))

    def forward(self, glob_s_emb: torch.Tensor, h_crit_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.model_dtype if self.model_dtype is not None else glob_s_emb.dtype
        
        projected_state = self.global_state_projector(glob_s_emb.to(target_dtype))
        
        rnn_weight_dtype = self.rnn.weight_ih_l0.dtype
        rnn_output, h_crit_next_unsq = self.rnn(
            projected_state.unsqueeze(1).to(rnn_weight_dtype), 
            h_crit_prev.unsqueeze(0).to(rnn_weight_dtype)
        )
        
        value_prediction = self.value_head(rnn_output.squeeze(1).to(target_dtype))
        return value_prediction.squeeze(-1), h_crit_next_unsq.squeeze(0).to(target_dtype)

# --- Shared LLM Loading Function ---
def load_shared_llm_and_tokenizer(global_config: GlobalSHPPOConfig, device: torch.device) -> Optional[Dict[str, Any]]:
    if not global_config.llm_model_name:
        logger.warning("No LLM model name specified in global_config. Skipping LLM loading.")
        return None
    logger.info(f"Loading shared LLM: {global_config.llm_model_name} for all roles on device: {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(global_config.llm_model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        if global_config.model_dtype is not None:
            compute_dtype = global_config.model_dtype
        elif device.type == 'cuda' and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float32
        logger.info(f"LLM compute_dtype determined as: {compute_dtype}")
        
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        device_map_value: Union[str, Dict[str, Any]] = "auto"
        if device.type == 'cpu': device_map_value = 'cpu'

        llm_base = AutoModelForCausalLM.from_pretrained(
            global_config.llm_model_name, 
            quantization_config=bnb_cfg if device.type == 'cuda' else None,
            device_map=device_map_value,
            torch_dtype=compute_dtype, 
            low_cpu_mem_usage=True if device.type == 'cuda' else False,
            trust_remote_code=True,
        )
        
        global_config.update_llm_dims(llm_base.config.hidden_size, compute_dtype)

        lora_cfg = LoraConfig(
            r=global_config.lora_r, lora_alpha=global_config.lora_alpha,
            target_modules=global_config.lora_target_modules,
            lora_dropout=global_config.lora_dropout, bias="none", task_type="CAUSAL_LM"
        )
        llm_peft_model = get_peft_model(llm_base, lora_cfg)
        llm_peft_model.to(device) 
        llm_lora_params = [p for n, p in llm_peft_model.named_parameters() if "lora_" in n and p.requires_grad]
        logger.info(f"Shared LLM ({global_config.llm_model_name}) loaded with LoRA. Trainable LoRA params: {sum(p.numel() for p in llm_lora_params if p.requires_grad)}. LLM dtype: {llm_peft_model.dtype}")
        return {"llm_model": llm_peft_model, "tokenizer": tokenizer, "llm_lora_params": llm_lora_params}
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}", exc_info=True)
        raise RuntimeError(f"LLM loading failed: {e}")

# --- Network Build Function ---
def build_actor_latent_for_role(
    global_config: GlobalSHPPOConfig,
    role_config: Union[PlannerConfig, CoderConfig, DebuggerConfig], 
    device: torch.device
) -> Dict[str, nn.Module]:
    role_name = role_config.role_name
    logger.info(f"Building ActorNet and LatentNet for role: {role_name} (obs_embed_dim: {role_config.obs_embed_dim}, N_ACTION_TEMPLATES: {role_config.N_ACTION_TEMPLATES}) on device: {device}")
    if role_config.obs_embed_dim <=0 or role_config.N_ACTION_TEMPLATES <=0:
        raise ValueError(f"RoleConfig for {role_name} is not properly initialized.")

    model_dtype_to_use = global_config.model_dtype if global_config.model_dtype is not None else torch.float32
    logger.info(f"Using dtype {model_dtype_to_use} for role {role_name} networks.")

    encoder_module = Encoder(role_config, global_config).to(device, dtype=model_dtype_to_use)
    latent_net_module = LatentNet(encoder_module, role_config, global_config).to(device, dtype=model_dtype_to_use)
    actor_net_module = ActorNet(role_config, global_config).to(device, dtype=model_dtype_to_use)
    return {"actor_net": actor_net_module, "latent_net": latent_net_module}

def build_networks(config: GlobalSHPPOConfig, device: torch.device) -> Dict[str, Any]:
    networks = {}
    trainable_params: Dict[str, List[torch.nn.Parameter]] = {
        "llm_lora": [], "actor_core": [], "latent": [], "critic": [], "inference": []
    }

    llm_info = load_shared_llm_and_tokenizer(config, device) 
    if llm_info:
        networks["llm_model"] = llm_info["llm_model"]
        networks["tokenizer"] = llm_info["tokenizer"]
        trainable_params["llm_lora"].extend(llm_info["llm_lora_params"])
    else:
        raise RuntimeError("LLM loading failed, cannot proceed.")

    networks["role_networks"] = {} 
    if config.model_dtype is None:
        raise RuntimeError("config.model_dtype was not set after LLM loading. This is a bug.")

    if config.planner_role_config and config.total_planner_agents > 0 :
        planner_nets = build_actor_latent_for_role(config, config.planner_role_config, device)
        networks["role_networks"]["planner"] = planner_nets
        trainable_params["actor_core"].extend(list(planner_nets["actor_net"].parameters()))
        trainable_params["latent"].extend(list(planner_nets["latent_net"].parameters()))
    if config.coder_role_config and config.total_coder_agents > 0:
        coder_nets = build_actor_latent_for_role(config, config.coder_role_config, device)
        networks["role_networks"]["coder"] = coder_nets
        trainable_params["actor_core"].extend(list(coder_nets["actor_net"].parameters()))
        trainable_params["latent"].extend(list(coder_nets["latent_net"].parameters()))
    if config.debugger_role_config and config.total_debugger_agents > 0:
        debugger_nets = build_actor_latent_for_role(config, config.debugger_role_config, device)
        networks["role_networks"]["debugger"] = debugger_nets
        trainable_params["actor_core"].extend(list(debugger_nets["actor_net"].parameters()))
        trainable_params["latent"].extend(list(debugger_nets["latent_net"].parameters()))

    model_dtype_to_use = config.model_dtype # 이미 설정됨
    logger.info(f"Building unified CriticNet (global_state_dim: {config.global_state_dim}) on device: {device} with dtype: {model_dtype_to_use}")
    critic_net_module = CriticNet(config).to(device, dtype=model_dtype_to_use)
    networks["critic_net"] = critic_net_module
    trainable_params["critic"].extend(list(critic_net_module.parameters()))

    logger.info(f"Building unified InferenceNet (global_state_dim: {config.global_state_dim}) on device: {device} with dtype: {model_dtype_to_use}")
    inference_net_module = InferenceNet(config).to(device, dtype=model_dtype_to_use)
    networks["inference_net"] = inference_net_module
    trainable_params["inference"].extend(list(inference_net_module.parameters()))

    networks["params"] = trainable_params
    return networks

def cosine_diversity(z_roles_batch: torch.Tensor) -> torch.Tensor:
    batch_size, num_agents, latent_dim = z_roles_batch.shape
    if num_agents <= 1:
        return torch.tensor(0.0, device=z_roles_batch.device, dtype=z_roles_batch.dtype)
    z_normalized = F.normalize(z_roles_batch.to(torch.float32), p=2, dim=-1)
    similarity_matrix = torch.bmm(z_normalized, z_normalized.transpose(1, 2)) 
    indices_i = torch.triu_indices(num_agents, num_agents, offset=1, device=z_roles_batch.device)
    masked_similarities = similarity_matrix[:, indices_i[0], indices_i[1]]
    if masked_similarities.numel() == 0:
        return torch.tensor(0.0, device=z_roles_batch.device, dtype=z_roles_batch.dtype)
    distances = 1.0 - masked_similarities
    diversity = distances.mean()
    return diversity.to(z_roles_batch.dtype) 

class SHPPOTrainer:
    def __init__(self, cfg: GlobalSHPPOConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        if self.cfg.model_dtype is None: 
             self.cfg.model_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
             logger.warning(f"GlobalSHPPOConfig.model_dtype was not set, defaulting to {self.cfg.model_dtype}")
        network_components = build_networks(self.cfg, self.device)
        self.llm_model = network_components["llm_model"]
        self.tokenizer = network_components["tokenizer"]
        self.role_networks = network_components["role_networks"] 
        self.critic_net = network_components["critic_net"]
        self.inference_net = network_components["inference_net"]
        self.all_trainable_parameters_grouped = network_components["params"]
        self.trainable_params_flat_list = []
        for param_group_name, param_list in self.all_trainable_parameters_grouped.items():
            if param_list: 
                self.trainable_params_flat_list.extend(param_list)
        if not self.trainable_params_flat_list:
            logger.warning("No trainable parameters found.")
        actor_core_params, latent_params = [], []
        critic_params    = list(self.critic_net.parameters())
        infer_params     = list(self.inference_net.parameters())
        lora_params      = self.all_trainable_parameters_grouped["llm_lora"]

        for role, nets in self.role_networks.items():
            actor_core_params.extend(nets["actor_net"].parameters())
            latent_params    .extend(nets["latent_net"].parameters())

        self.opt_actor     = optim.AdamW(list(actor_core_params) + list(lora_params),
                                        lr=cfg.lr_actor, betas=(0.9, 0.999))
        self.opt_critic    = optim.AdamW(critic_params, lr=cfg.lr_critic)
        self.opt_latent    = optim.AdamW(latent_params, lr=cfg.lr_latent)
        self.opt_inference = optim.AdamW(infer_params, lr=cfg.lr_infer)

        
        self.optimizer = self.opt_actor
        self.train_dataset = CodeContestDataset(split="train", max_problems=-1, max_cases=3)
        self.eval_dataset = CodeContestDataset(split="valid", max_problems=-1, max_cases=-1)
        self.test_dataset = CodeContestDataset(split="test", max_problems=-1, max_cases=-1)
        
        from marcog_test import CodeTester 
        self.code_tester = CodeTester(
            dataset=self.train_dataset, global_config=self.cfg,
            tokenizer=self.tokenizer, llm_model=self.llm_model,
            log_file=os.environ.get("EXECUTION_LOG_FILE", f"execution_log_train_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv")
        )
        self.eval_tester = CodeTester(
            dataset=self.eval_dataset, global_config=self.cfg,
            tokenizer=self.tokenizer, llm_model=self.llm_model,
            log_file=f"execution_log_eval_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv"
        )
        self.test_tester = CodeTester(
            dataset=self.test_dataset, global_config=self.cfg,
            tokenizer=self.tokenizer, llm_model=self.llm_model,
            log_file=f"execution_log_test_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv"
        )
        logger.info(f"Loaded {len(self.train_dataset.get_all_tasks())} tasks for training, {len(self.eval_dataset.get_all_tasks())} for validation, {len(self.test_dataset.get_all_tasks())} for test.")

    def update_learning_rate(self, current_update: int):
        new_lr = self.cfg.get_learning_rate(current_update)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def sample_problems(self, num_problems: int) -> List[Dict[str, Any]]:
        all_tasks = self.train_dataset.get_all_tasks()
        if not all_tasks: return []
        return random.sample(all_tasks, min(num_problems, len(all_tasks)))

    def collect_rollouts(self) -> List[Dict[str, Any]]:
        problems_for_batch = self.sample_problems(self.cfg.num_problems_per_batch)
        if not problems_for_batch:
            logger.warning("No problems sampled for rollout collection.")
            return []
        all_problem_data_for_batch: List[Dict[str, Any]] = []
        for problem_idx, task in enumerate(tqdm(problems_for_batch, desc="Collecting Rollouts", leave=False)):
            initial_h_critic_for_problem = torch.zeros(1, self.cfg.critic_rnn_hidden_dim, device=self.device, dtype=self.cfg.model_dtype)
            collected_data_for_this_problem = self.code_tester.run_pipeline_for_problem(
                problem_idx, task, self.tokenizer, self.llm_model,
                self.role_networks, self.critic_net, self.inference_net,
                initial_h_critic_for_problem
            )
            all_problem_data_for_batch.append(collected_data_for_this_problem)
        return all_problem_data_for_batch

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        advantages = rewards - values 
        if self.cfg.use_gae_normalization and advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def shppo_update(self,
                     batch_of_problem_data: List[Dict[str, Any]]
                    ) -> Dict[str, float]:

        if not batch_of_problem_data:
            logger.warning("empty batch"); return {}

        cfg, dev        = self.cfg, self.device
        f32             = torch.float32
        bf              = cfg.model_dtype or torch.bfloat16   

        
        R   = torch.tensor([d["final_reward"]
                            for d in batch_of_problem_data],
                           dtype=f32, device=dev)                             
        V0  = torch.tensor([d["initial_value_prediction"]
                            for d in batch_of_problem_data],
                           dtype=bf,  device=dev)                             

        Sg  = torch.stack([d["initial_global_state_embedding"]
                           for d in batch_of_problem_data]).squeeze(1)
        Sg  = Sg.to(dev, dtype=bf)

        Hc0 = torch.stack([d["initial_critic_hidden_state"]
                           for d in batch_of_problem_data]).squeeze(1)
        Hc0 = Hc0.to(dev, dtype=bf)

        adv_per_problem = self.compute_advantages(R, V0.to(f32))

        turn2prob: list[int] = []                                             
        flat = {
            r: {k: [] for k in ("obs", "h", "act", "oldlp", "adv")}
            for r in self.role_networks
        }

        for p_idx, pdata in enumerate(batch_of_problem_data):
            for role, nets in self.role_networks.items():
                t = pdata["actor_inputs_by_role"].get(role)
                if t is None or t["obs_embs"].numel() == 0:
                    continue

                n_turn = t["old_log_probs"].size(0)
                flat[role]["obs"  ].append(t["obs_embs"].to(bf))
                flat[role]["h"    ].append(t["h_prevs"  ].to(bf))
                flat[role]["act"  ].append(t["chosen_actions"].long())
                flat[role]["oldlp"].append(t["old_log_probs"].to(f32))
                flat[role]["adv"  ].append(
                    adv_per_problem[p_idx].repeat(n_turn).to(f32)
                )
                turn2prob.extend([p_idx] * n_turn)

        for role in flat:
            for k in ("obs", "h"):
                flat[role][k] = (torch.cat(flat[role][k]).to(dev, dtype=bf)
                                 if flat[role][k] else
                                 torch.empty(0, device=dev, dtype=bf))
            flat[role]["act"]   = (torch.cat(flat[role]["act"]).to(dev) 
                                   if flat[role]["act"] else
                                   torch.empty(0, device=dev, dtype=torch.long))
            flat[role]["oldlp"] = (torch.cat(flat[role]["oldlp"]).to(dev, dtype=f32)
                                   if flat[role]["oldlp"] else
                                   torch.empty(0, device=dev, dtype=f32))
            flat[role]["adv"]   = (torch.cat(flat[role]["adv"]).to(dev, dtype=f32)
                                   if flat[role]["adv"] else
                                   torch.empty(0, device=dev, dtype=f32))

        if all(flat[r]["obs"].numel() == 0 for r in flat):
            return {}   

        all_adv = torch.cat([flat[r]["adv"] for r in flat])
        if all_adv.numel() > 1:
            std = all_adv.std(unbiased=False)
            if std > 1e-8:
                all_adv = (all_adv - all_adv.mean()) / (std + 1e-8)
            else:
                all_adv = all_adv - all_adv.mean()
        else:
            all_adv = all_adv - all_adv.mean()

        offset = 0
        for role in flat:
            n = flat[role]["adv"].size(0)
            flat[role]["adv"] = all_adv[offset:offset + n]
            offset += n

        turn2prob_t = torch.as_tensor(turn2prob, device=dev)

        stats = dict(actor=0., entropy=0., critic=0.,
                     latent=0., infer=0.)
        nA = nC = nL = nI = 0

        # ========================= 1) ACTOR =========================
        for role, nets in self.role_networks.items():
            if flat[role]["obs"].numel() == 0:
                continue

            actnet, latnet = nets["actor_net"], nets["latent_net"]
            O, H, A, OLP, ADV = (flat[role][k] for k in
                                 ("obs", "h", "act", "oldlp", "adv"))

            M     = O.size(0)
            bsize = max(1, M // cfg.num_minibatches)
            perm  = torch.randperm(M, device=dev)

            for s in range(0, M, bsize):
                mb  = perm[s:s + bsize]
                o_b = O[mb].detach()
                h_b = H[mb].detach()

                with torch.no_grad():
                    z_b, _, _ = latnet(o_b, h_b)  

                logits_b, *_ = actnet(o_b, h_b, z_b)  
                dist        = Categorical(logits=logits_b.to(f32))

                ratio = torch.exp(dist.log_prob(A[mb]) - OLP[mb])
                surr1 = ratio * ADV[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * ADV[mb]
                lossA = -torch.min(surr1, surr2).mean() \
                        - cfg.ent_coef_actor * dist.entropy().mean()

                self.opt_actor.zero_grad()
                lossA.backward()
                nA += 1

                stats["actor"]   += lossA.item()
                stats["entropy"] += dist.entropy().mean().item()

            torch.nn.utils.clip_grad_norm_(actnet.parameters(), cfg.max_grad_norm)
            self.opt_actor.step()

        # ========================= 2) CRITIC =========================
        B      = R.size(0)
        bsizeT = max(1, B // cfg.num_minibatches)
        permT  = torch.randperm(B, device=dev)

        for s in range(0, B, bsizeT):
            mb     = permT[s:s + bsizeT]
            # critic 입력도 BF16
            v_pred, _ = self.critic_net(Sg[mb], Hc0[mb])  
            target    = R[mb].to(v_pred.dtype)          

            v_pred_flat = v_pred.view(-1)
            target_flat = target.view(-1)

            diff   = (v_pred_flat.float() - target_flat.float())**2
            lossV  = diff.mean().to(v_pred.dtype) * cfg.vf_coef

            self.opt_critic.zero_grad()
            lossV.backward()
            nC += 1

        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(),
                                       cfg.max_grad_norm)
        self.opt_critic.step()
        stats["critic"] += lossV.item()

        # ================ 3) LATENT  &  4) INFERENCE ================
        for role, nets in self.role_networks.items():
            if flat[role]["obs"].numel() == 0:
                continue

            latnet = nets["latent_net"]
            O, H = flat[role]["obs"], flat[role]["h"]

            M     = O.size(0)
            bsize = max(1, M // cfg.num_minibatches)
            perm  = torch.randperm(M, device=dev)

            for s in range(0, M, bsize):
                mb   = perm[s:s + bsize]
                o_b  = O[mb] 
                h_b  = H[mb] 
                gidx = turn2prob_t[mb]  

                z, mu, sig = latnet(o_b, h_b)  
                sig32 = sig.float().clamp(min=1e-6)  


                entropy_lat = (
                    0.5 * cfg.latent_dim * (1 + math.log(2 * math.pi))
                    + torch.log(sig32).sum(-1)
                ).mean()  

                diversity_lat = cosine_diversity(z.mean(1).unsqueeze(0)).to(f32)

                V_I_lat = self.inference_net(
                    Sg[gidx].to(bf),  
                    mu.reshape(len(mb), -1).to(bf),
                    sig.reshape(len(mb), -1).to(bf)
                )  

                
                lossL_fp32 = (
                    -cfg.lambda_V_I_guidance_for_latent * V_I_lat.float().mean()
                    + cfg.lambda_entropy_latent   * entropy_lat
                    - cfg.lambda_diversity_latent * diversity_lat
                )

                lossL = lossL_fp32.to(bf)

                self.opt_latent.zero_grad()
                lossL.backward()
                nL += 1

                torch.nn.utils.clip_grad_norm_(latnet.parameters(),
                                               cfg.max_grad_norm)
                self.opt_latent.step()
                stats["latent"] += lossL.item()

                
                mu_det  = mu.detach().reshape(len(mb), -1).to(bf)
                sig_det = sig.detach().reshape(len(mb), -1).to(bf)

                V_I2    = self.inference_net(
                    Sg[gidx].to(bf),  
                    mu_det,           
                    sig_det           
                )  

                target_bf = R[gidx].to(V_I2.dtype)

                loss_inf_f32 = F.mse_loss(V_I2.float(), target_bf.float())
                loss_inf = (loss_inf_f32 * cfg.lambda_inf_mse).to(V_I2.dtype)

                self.opt_inference.zero_grad()
                loss_inf.backward()
                torch.nn.utils.clip_grad_norm_(self.inference_net.parameters(), cfg.max_grad_norm)
                self.opt_inference.step()
                stats["infer"] += loss_inf.item(); nI += 1

        if nA:
            stats["actor"]   /= nA
            stats["entropy"] /= nA
        if nC: stats["critic"] /= nC
        if nL: stats["latent"] /= nL
        if nI: stats["infer"]  /= nI

        return stats



    def train(self, updates: Optional[int] = None):
        num_updates = updates if updates is not None else self.cfg.updates
        if wandb.run is None:
            wandb.init(
                project=self.cfg.wandb_project_name,
                config=vars(self.cfg),
                name=f"{self.cfg.wandb_run_name_prefix}-{num_updates}Updates-{pd.Timestamp.now().strftime('%Y%m%d%H%M')}",
                reinit=True
            )
        num_trainable_total = sum(p.numel() for p in self.trainable_params_flat_list)
        logger.info(f"Total trainable parameters (RL modules + LoRA): {num_trainable_total}")
        if "llm_lora" in self.all_trainable_parameters_grouped:
             logger.info(f"  - Trainable LoRA parameters: {sum(p.numel() for p in self.all_trainable_parameters_grouped['llm_lora'])}")
        wandb.config.update({"num_trainable_params_total": num_trainable_total}, allow_val_change=True)
        logger.info(f"Starting SHPPO training for {num_updates} updates...")
        best_avg_reward_train = -float('inf')
        best_val_pass_rate = -float('inf')
        for update_idx in range(1, num_updates + 1):
            current_lr = self.update_learning_rate(update_idx -1)
            rollout_batch_data = self.collect_rollouts()
            if not rollout_batch_data:
                logger.warning(f"Update {update_idx}: No rollouts collected, skipping update step.")
                continue
            loss_metrics = self.shppo_update(batch_of_problem_data=rollout_batch_data)
            current_batch_rewards = torch.stack([d["final_reward"] for d in rollout_batch_data])
            avg_reward_train = current_batch_rewards.mean().item()
            max_reward_train = current_batch_rewards.max().item()
            min_reward_train = current_batch_rewards.min().item()
            if avg_reward_train > best_avg_reward_train: best_avg_reward_train = avg_reward_train
            log_dict = {
                "update": update_idx, "train/learning_rate": current_lr,
                "train/avg_reward": avg_reward_train, "train/max_reward": max_reward_train,
                "train/min_reward": min_reward_train, "train/best_avg_reward": best_avg_reward_train,
                **{f"loss/{k}": v for k,v in loss_metrics.items()}
            }
            if update_idx % self.cfg.log_interval == 0:
                log_str = f"[Update {update_idx:04d}/{num_updates}] LR={current_lr:.2e}, AvgR={avg_reward_train:.3f}"
                for k,v in loss_metrics.items(): log_str += f", {k}={v:.3f}"
                logger.info(log_str)
            if update_idx % self.cfg.evaluate_interval == 0 or update_idx == num_updates:
                logger.info(f"\n--- Validation Evaluation at Update {update_idx} ---")
                for role_name_key_eval in self.role_networks:
                    self.role_networks[role_name_key_eval]['actor_net'].eval()
                    self.role_networks[role_name_key_eval]['latent_net'].eval()
                self.critic_net.eval(); self.inference_net.eval()
                if hasattr(self.llm_model, 'eval'): self.llm_model.eval()
                try:
                    val_pass_rate, _, _ = self.eval_tester.run(
                        tokenizer=self.tokenizer, llm_model=self.llm_model,
                        role_networks=self.role_networks, critic_net=self.critic_net,
                        inference_net=self.inference_net, log=False,
                        num_problems_to_run=self.cfg.num_problems_per_batch * 2
                    )
                    logger.info(f"Validation pass rate at update {update_idx}: {val_pass_rate:.2f}%")
                    if val_pass_rate > best_val_pass_rate:
                        best_val_pass_rate = val_pass_rate
                        logger.info(f"New best validation pass rate: {best_val_pass_rate:.2f}%")
                    log_dict["val/pass_rate"] = val_pass_rate
                    log_dict["val/best_pass_rate"] = best_val_pass_rate
                except Exception as e:
                    logger.error(f"Validation evaluation failed at update {update_idx}: {e}", exc_info=True)
                logger.info("--- End Validation Evaluation ---\n")
            wandb.log(log_dict, step=update_idx)
        logger.info("\n--- Final Test Evaluation ---")
        for role_name_key_eval_final in self.role_networks:
            self.role_networks[role_name_key_eval_final]['actor_net'].eval()
            self.role_networks[role_name_key_eval_final]['latent_net'].eval()
        self.critic_net.eval(); self.inference_net.eval()
        if hasattr(self.llm_model, 'eval'): self.llm_model.eval()
        try:
            test_pass_rate, _, _ = self.test_tester.run(
                tokenizer=self.tokenizer, llm_model=self.llm_model,
                role_networks=self.role_networks, critic_net=self.critic_net,
                inference_net=self.inference_net, log=True, num_problems_to_run=-1
            )
            logger.info(f"Final Test pass rate: {test_pass_rate:.2f}%")
            wandb.summary["final_test_pass_rate"] = test_pass_rate
        except Exception as e:
            logger.error(f"Final test evaluation failed: {e}", exc_info=True)
        wandb.summary["best_train_avg_reward"] = best_avg_reward_train
        wandb.summary["best_val_pass_rate"] = best_val_pass_rate
        logger.info("Training complete!")
        logger.info(f"Best training average reward: {best_avg_reward_train:.4f}")
        logger.info(f"Best validation pass rate: {best_val_pass_rate:.2f}%")
        wandb.finish()

    def full_evaluation(self, dataset_to_eval="test"):
        logger.info(f"Running full evaluation on {dataset_to_eval} split...")
        eval_target_tester = None
        if dataset_to_eval == "valid": eval_target_tester = self.eval_tester
        elif dataset_to_eval == "test": eval_target_tester = self.test_tester
        elif dataset_to_eval == "train": eval_target_tester = self.code_tester
        else: logger.error(f"Unknown dataset split for full_evaluation: {dataset_to_eval}"); return
        for role_name_key_eval_full in self.role_networks:
            self.role_networks[role_name_key_eval_full]['actor_net'].eval()
            self.role_networks[role_name_key_eval_full]['latent_net'].eval()
        self.critic_net.eval(); self.inference_net.eval()
        if hasattr(self.llm_model, 'eval'): self.llm_model.eval()
        try:
            pass_rate, _, _ = eval_target_tester.run(
                tokenizer=self.tokenizer, llm_model=self.llm_model,
                role_networks=self.role_networks, critic_net=self.critic_net,
                inference_net=self.inference_net, log=True, num_problems_to_run=-1
            )
            logger.info(f"Full evaluation on {dataset_to_eval} split - Pass rate: {pass_rate:.2f}%")
            if wandb.run: wandb.summary[f"full_eval_{dataset_to_eval}_pass_rate"] = pass_rate
        except Exception as e:
            logger.error(f"Full evaluation on {dataset_to_eval} split failed: {e}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cur_time_str = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    os.environ["EXECUTION_LOG_FILE"] = f"execution_log_main_train_{cur_time_str}.csv"
    torch.set_grad_enabled(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    cfg = GlobalSHPPOConfig()
    logger.info(f"Configured number of agents:")
    logger.info(f"  Planners: {cfg.total_planner_agents}")
    logger.info(f"  Coders: {cfg.total_coder_agents}")
    logger.info(f"  Debuggers: {cfg.total_debugger_agents}")
    cfg.model_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    cfg.device = device 
    try:
        trainer = SHPPOTrainer(cfg, device)
        trainer.train()
    except Exception as e:
        logger.error("Training failed with an exception.", exc_info=True)
        if wandb.run: wandb.finish(exit_code=1)
        raise