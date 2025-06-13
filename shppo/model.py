"""
model.py - Fixed SHPPO Model Implementation
LLM gradient issue 해결: embedding gradient와 LLM parameter update 분리
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# ══════════════════ 1. Configs ══════════════════
@dataclass
class RoleConfig:
    role_name: str
    obs_embed_dim: int
    n_action_templates: int

@dataclass
class SHPPOModelConfig:
    base_model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_r: int = 8
    lora_alpha: int = 16
    latent_dim: int = 16
    hete_layer_input_dim: int = 512
    hete_layer_output_dim: int = 64
    mlp_hidden_dim: int = 256
    dtype: torch.dtype | None = None

# ══════════════════ 2. Building blocks ══════════════════
class MLP(nn.Sequential):
    def __init__(self, inp: int, out: int, hid: int, n: int = 2):
        layers = []
        cur = inp
        for _ in range(n - 1):
            layers += [nn.Linear(cur, hid), nn.ReLU()]
            cur = hid
        layers.append(nn.Linear(cur, out))
        super().__init__(*layers)
        self.apply(lambda m: self._ortho(m))

    @staticmethod
    def _ortho(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, math.sqrt(2))
            nn.init.zeros_(m.bias)

class Encoder(nn.Module):
    """Latent Network Encoder - 논문의 LatentNet"""
    def __init__(self, rcfg: RoleConfig, cfg: SHPPOModelConfig):
        super().__init__()
        self.n_tpl = rcfg.n_action_templates
        self.latent_dim = cfg.latent_dim
        self.mlp = MLP(rcfg.obs_embed_dim + cfg.hete_layer_input_dim,
                       cfg.mlp_hidden_dim, cfg.mlp_hidden_dim)
        self.fc_mu  = nn.Linear(cfg.mlp_hidden_dim, self.n_tpl * self.latent_dim)
        self.fc_sig = nn.Linear(cfg.mlp_hidden_dim, self.n_tpl * self.latent_dim)
        for layer in (self.fc_mu, self.fc_sig):
            nn.init.orthogonal_(layer.weight, 0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor, h_prev: torch.Tensor):
        hid = self.mlp(torch.cat([obs, h_prev], dim=-1))
        mu  = self.fc_mu(hid)
        sig = F.softplus(self.fc_sig(hid)) + 1e-5
        B = mu.size(0)
        mu = mu.view(B, self.n_tpl, self.latent_dim)
        sig = sig.view(B, self.n_tpl, self.latent_dim)
        latent = mu + sig * torch.randn_like(sig)  # Reparameterization trick
        return latent, mu, sig

class TemplateDecoder(nn.Module):
    """Heterogeneous Layer Parameter Generator"""
    def __init__(self, cfg: SHPPOModelConfig):
        super().__init__()
        D_in, D_out = cfg.hete_layer_input_dim, cfg.hete_layer_output_dim
        self.fc_w = nn.Linear(cfg.latent_dim, D_in * D_out)
        self.fc_b = nn.Linear(cfg.latent_dim, D_out)
        for layer in (self.fc_w, self.fc_b):
            nn.init.orthogonal_(layer.weight, math.sqrt(0.1))
            nn.init.zeros_(layer.bias)
        self.D_out, self.D_in = D_out, D_in

    def forward(self, z_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W = self.fc_w(z_flat).view(-1, self.D_out, self.D_in)
        b = self.fc_b(z_flat)
        return W, b

class ActorHead(nn.Module):
    """Actor Head with Heterogeneous Layer"""
    def __init__(self, rcfg: RoleConfig, cfg: SHPPOModelConfig):
        super().__init__()
        self.decoder = TemplateDecoder(cfg)
        self.post = MLP(cfg.hete_layer_output_dim,
                        cfg.mlp_hidden_dim, cfg.mlp_hidden_dim, 1)
        self.logit = nn.Linear(cfg.mlp_hidden_dim, 1)
        nn.init.orthogonal_(self.logit.weight, 0.01)
        nn.init.zeros_(self.logit.bias)
        self.n_tpl = rcfg.n_action_templates

    def forward(self, h_t: torch.Tensor, lat_all: torch.Tensor) -> torch.Tensor:
        B, Tpl, Dz = lat_all.shape
        # Expand hidden state
        h_rep = h_t.unsqueeze(1).expand(-1, Tpl, -1).reshape(-1, h_t.size(-1))
        W, b = self.decoder(lat_all.reshape(-1, Dz))
        hete = torch.bmm(h_rep.unsqueeze(1), W.transpose(1,2)).squeeze(1) + b
        feat = self.post(hete)
        return self.logit(feat).view(B, Tpl)

class MultiAgentActor(nn.Module):
    """Multi-Agent Actor with Latent Learning"""
    def __init__(self, rcfg: RoleConfig, cfg: SHPPOModelConfig):
        super().__init__()
        self.enc = Encoder(rcfg, cfg)
        self.gru = nn.GRU(cfg.hete_layer_input_dim,
                          cfg.hete_layer_input_dim,
                          batch_first=True)
        self.head = ActorHead(rcfg, cfg)

    def forward(self, obs_emb: torch.Tensor, h_prev: torch.Tensor) -> tuple[Any, Any, Any]:
        h_next = self.gru(obs_emb.unsqueeze(1), h_prev.unsqueeze(0))[1].squeeze(0)
        latent, mu, sig = self.enc(obs_emb, h_prev)
        logits = self.head(h_next, latent)
        return logits, h_next, (latent, mu, sig)

class MultiHeadCritic(nn.Module):
    """Multi-Head Critic for Different Roles"""
    def __init__(self, hidden_dim: int, roles: List[str]):
        super().__init__()
        self.role = nn.ModuleDict({r: nn.Linear(hidden_dim,1) for r in roles})
        self.global_v = nn.Linear(hidden_dim,1)
        for lin in list(self.role.values()) + [self.global_v]:
            nn.init.orthogonal_(lin.weight, 1.0)
            nn.init.zeros_(lin.bias)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {r: head(h).squeeze(-1) for r, head in self.role.items()}
        out['global'] = self.global_v(h).squeeze(-1)
        return out

# ══════════════════ 3. Enhanced Inference Network (Scalable) ══════════════════
class ScalableInferenceNet(nn.Module):
    """Scalable InferenceNet that handles variable number of agents using attention"""
    
    def __init__(self, cfg: SHPPOModelConfig):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.hidden_dim = cfg.mlp_hidden_dim
        
        # Agent feature encoder (processes mu + sigma for each agent)
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Multi-head attention for scalable aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # Global observation encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(cfg.hete_layer_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Value prediction head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, math.sqrt(2))
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        agent_latent_features: torch.Tensor,  # [batch_size, n_agents, feature_dim]
        global_obs: torch.Tensor  # [batch_size, global_obs_dim]
    ) -> torch.Tensor:
        """
        Forward pass with attention-based aggregation for scalability
        
        Args:
            agent_latent_features: [batch_size, n_agents, latent_dim*2] (mu+sigma concatenated)
            global_obs: [batch_size, global_obs_dim]
        Returns:
            value: [batch_size] - predicted value for the team
        """
        batch_size = global_obs.size(0)
        n_agents = agent_latent_features.size(1)
        
        # Encode each agent's latent features: [batch_size, n_agents, hidden_dim]
        agent_encoded = self.agent_encoder(agent_latent_features)
        
        # Self-attention aggregation (handles variable n_agents)
        aggregated_agents, attention_weights = self.attention(
            agent_encoded, agent_encoded, agent_encoded
        )
        
        # Pool across agents: [batch_size, hidden_dim]
        pooled_agents = torch.mean(aggregated_agents, dim=1)
        
        # Encode global observation: [batch_size, hidden_dim]
        global_features = self.global_encoder(global_obs)
        
        # Combine agent and global features: [batch_size, hidden_dim * 2]
        combined = torch.cat([pooled_agents, global_features], dim=-1)
        
        # Predict value: [batch_size]
        value = self.value_head(combined).squeeze(-1)
        
        return value

# ══════════════════ 4. Complete Loss Computer ══════════════════
class SHPPOLossComputer:
    """Complete implementation of all SHPPO losses from the paper"""
    
    def __init__(self, lambda_e: float = 0.01, lambda_d: float = 0.1):
        self.lambda_e = lambda_e  # Entropy loss weight (논문 Equation 10)
        self.lambda_d = lambda_d  # Distance loss weight (논문 Equation 10)
    
    def compute_entropy_loss(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy loss Le = (1/n) * Σ H(Di) (논문 Equation 7)
        
        Args:
            mu: [batch_size, n_templates, latent_dim] - mean of latent distributions
            sigma: [batch_size, n_templates, latent_dim] - std of latent distributions
        Returns:
            entropy_loss: scalar tensor
        """
        # Entropy of multivariate Gaussian with diagonal covariance:
        # H(X) = 0.5 * log((2πe)^k * |Σ|) = 0.5 * Σ log(2πe * σ²)
        log_sigma_sq = 2 * torch.log(sigma + 1e-8)
        entropy_per_agent = 0.5 * torch.sum(
            math.log(2 * math.pi * math.e) + log_sigma_sq, 
            dim=-1  # Sum over latent dimensions
        )
        # Average over templates and batch
        return entropy_per_agent.mean()
    
    def compute_distance_loss(self, latent_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute distance loss Ld (논문 Equation 8)
        
        Args:
            latent_samples: [batch_size, n_templates, latent_dim] - sampled latent variables
        Returns:
            distance_loss: scalar tensor
        """
        batch_size, n_templates, latent_dim = latent_samples.shape
        
        if n_templates < 2:
            return torch.tensor(0.0, device=latent_samples.device)
        
        # Flatten latent for cosine similarity: [batch_size, n_templates, latent_dim]
        latent_flat = latent_samples.reshape(batch_size, n_templates, -1)
        
        distances = []
        
        # Compute pairwise distances between all template pairs
        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                # Get latent vectors for templates i and j
                li = latent_flat[:, i]  # [batch_size, latent_dim]
                lj = latent_flat[:, j]  # [batch_size, latent_dim]
                
                # Cosine similarity
                cos_sim = F.cosine_similarity(li, lj, dim=-1)  # [batch_size]
                
                # Distance: 1 - cosine_similarity
                distance = 1 - cos_sim  # [batch_size]
                distances.append(distance)
        
        if distances:
            # Stack all distances: [n_pairs, batch_size]
            all_distances = torch.stack(distances, dim=0)
            
            # Normalize distances according to Equation 9
            normalized = self._normalize_distances(all_distances)
            
            # Return mean distance
            return normalized.mean()
        else:
            return torch.tensor(0.0, device=latent_samples.device)
    
    def _normalize_distances(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Normalize distances according to 논문 Equation 9:
        Norm(xi) = (xi - min(x)) / (max(x) - min(x) + 1e-12)
        """
        min_dist = torch.min(distances)
        max_dist = torch.max(distances)
        
        # Avoid division by zero
        denominator = max_dist - min_dist + 1e-12
        
        if denominator < 1e-10:
            return distances
        
        normalized = (distances - min_dist) / denominator
        return normalized
    
    def compute_combined_latent_loss(
        self,
        value_estimate: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        latent_samples: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined latent loss LL according to 논문 Equation 10:
        LL(θL) = -Lv(θL) + λe*Le(θL) - λd*Ld(θL)
        
        Args:
            value_estimate: [batch_size] - output from InferenceNet
            mu: [batch_size, n_templates, latent_dim]
            sigma: [batch_size, n_templates, latent_dim] 
            latent_samples: [batch_size, n_templates, latent_dim]
        Returns:
            dict with all loss components
        """
        
        # 1. Value loss: Lv = VI (논문 Equation 6)
        value_loss = value_estimate.mean()
        
        # 2. Entropy loss: Le (논문 Equation 7)
        entropy_loss = self.compute_entropy_loss(mu, sigma)
        
        # 3. Distance loss: Ld (논문 Equation 8)
        distance_loss = self.compute_distance_loss(latent_samples)
        
        # 4. Combined latent loss: LL (논문 Equation 10)
        combined_loss = (
            -value_loss + 
            self.lambda_e * entropy_loss - 
            self.lambda_d * distance_loss
        )
        
        return {
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'distance_loss': distance_loss,
            'combined_latent_loss': combined_loss
        }

# ══════════════════ 5. FIXED Enhanced SharedLLM ══════════════════
class EnhancedSharedLLM(nn.Module):
    """Enhanced SharedLLM with simplified training control (always enabled)"""
    
    def __init__(self, cfg: SHPPOModelConfig, device: Union[str, torch.device]):
        super().__init__()
        
        # 4-bit quantization config
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            device_map='auto',
            quantization_config=bnb,
            trust_remote_code=True
        )
        
        # Add LoRA
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            bias='none',
            task_type='CAUSAL_LM'
        )
        
        self.model = get_peft_model(base, lora_cfg).to(device).eval()
        
        # SIMPLIFIED: Always enable LoRA training by default
        self._setup_trainable_parameters()
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        cfg.dtype = self.model.dtype
        self.cfg = cfg
        self.training_mode = True
        
        # Code generation templates conditioned on latent variables
        self.generation_templates = {
            "planner": [
                "# Greedy Algorithm Approach\n1. Sort input data\n2. Make greedy choices\n3. Verify optimality",
                "# Dynamic Programming Approach\n1. Define state space\n2. Find recurrence relation\n3. Build solution bottom-up",
                "# Graph Algorithm Approach\n1. Model as graph problem\n2. Apply appropriate traversal\n3. Extract path/solution",
                "# Mathematical Approach\n1. Identify mathematical pattern\n2. Derive closed-form solution\n3. Handle edge cases"
            ],
            "coder": [
                """def solve(input_str):
    # Greedy approach implementation
    lines = input_str.strip().split('\\n')
    data = [int(x) for x in lines[0].split()]
    data.sort()
    return str(sum(data))""",
                
                """def solve(input_str):
    # Dynamic programming implementation  
    n = int(input_str.strip().split()[0])
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i-1] + i
    return str(dp[n])""",
                
                """def solve(input_str):
    # Graph-based implementation
    from collections import defaultdict, deque
    lines = input_str.strip().split('\\n')
    n = int(lines[0])
    graph = defaultdict(list)
    # Build and process graph
    return str(n)""",
                
                """def solve(input_str):
    # Mathematical solution
    import math
    n = int(input_str.strip().split()[0])
    result = n * (n + 1) // 2
    return str(result)"""
            ],
            "debugger": [
                """def solve(input_str):
    # Robust greedy with error handling
    try:
        lines = input_str.strip().split('\\n')
        if not lines or not lines[0]:
            return "0"
        data = [int(x) for x in lines[0].split() if x.isdigit()]
        return str(sum(sorted(data)))
    except Exception:
        return "0" """,
                
                """def solve(input_str):
    # Robust DP with bounds checking
    try:
        n = int(input_str.strip().split()[0])
        if n < 0 or n > 10**6:
            return "0"
        result = n * (n + 1) // 2
        return str(result)
    except Exception:
        return "0" """,
                
                """def solve(input_str):
    # Robust graph solution
    try:
        from collections import defaultdict
        lines = input_str.strip().split('\\n')
        n = max(1, int(lines[0]) if lines and lines[0].isdigit() else 1)
        return str(min(n, 1000))
    except Exception:
        return "1" """,
                
                """def solve(input_str):
    # Robust mathematical solution
    try:
        import math
        n = int(input_str.strip().split()[0])
        result = max(0, min(n * (n + 1) // 2, 10**9))
        return str(result)
    except Exception:
        return "0" """
            ]
        }
    
    def _setup_trainable_parameters(self):
        """Setup trainable parameters - LoRA always enabled"""
        for name, param in self.model.named_parameters():
            if 'lora_' in name or 'Lora' in name:
                param.requires_grad = True  # Always enable LoRA training
            else:
                param.requires_grad = False
    
    def set_llm_training_mode(self, enabled: bool):
        """Set LLM training mode - simplified for always-on training"""
        if enabled:
            self._setup_trainable_parameters()
            print("🔥 LLM LoRA parameters enabled for training")
        else:
            # This shouldn't be called in the new design, but keep for compatibility
            for name, param in self.model.named_parameters():
                if 'lora_' in name or 'Lora' in name:
                    param.requires_grad = False
            print("❄️ LLM LoRA parameters frozen")
    
    def set_training_mode(self, mode: bool):
        """Control training mode"""
        self.training_mode = mode
    
    def embed_observation_with_gradient(
        self, 
        prompts: List[str],
        allow_grad: bool = True
    ) -> torch.Tensor:
        """
        Simplified: Embed observations with gradient computation
        
        Args:
            prompts: List of text prompts to embed
            allow_grad: Whether to allow gradients through computation graph
        Returns:
            embeddings: [batch_size, hidden_size]
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Simplified: Just compute with or without gradients
        if allow_grad:
            outputs = self.model(**inputs, output_hidden_states=True)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract embeddings from last hidden layer
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Average pooling over sequence length
        embeddings = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        return embeddings.float()
    
    def generate_with_latent_conditioning(
        self, 
        role: str, 
        action_idx: int, 
        latent: torch.Tensor
    ) -> str:
        """
        Generate code content conditioned on latent variables
        
        Args:
            role: Agent role ("planner", "coder", "debugger")
            action_idx: Action template index
            latent: Latent variable tensor [batch_size, n_templates, latent_dim]
        Returns:
            generated_content: String content for the file
        """
        if role not in self.generation_templates:
            role = "coder"  # fallback
        
        templates = self.generation_templates[role]
        
        # Use latent to select and modify template
        if latent.numel() > 0:
            # Convert latent to selection signal
            latent_flat = latent.flatten()
            selection_signal = torch.sum(latent_flat).item()
            
            # Map to template index
            template_idx = int(abs(selection_signal * 1000)) % len(templates)
        else:
            template_idx = action_idx % len(templates)
        
        base_template = templates[template_idx]
        
        # Add latent-based variations
        if latent.numel() > 0:
            latent_sum = torch.sum(latent).item()
            
            # Modify template based on latent values
            if role == "planner" and latent_sum > 0.5:
                base_template += "\n4. Optimize for performance\n5. Consider alternative approaches"
            elif role == "coder" and latent_sum < -0.5:
                # Add more robust error handling
                if "try:" not in base_template:
                    lines = base_template.split('\n')
                    # Wrap main logic in try-except
                    main_logic = '\n'.join(lines[1:])  # Skip def line
                    base_template = lines[0] + "\n    try:\n" + \
                                  '\n'.join("    " + line for line in lines[1:]) + \
                                  "\n    except Exception:\n        return '0'"
        
        return base_template
    
    def trainable_parameters(self) -> List[torch.Tensor]:
        """Return list of trainable LoRA parameters"""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    @staticmethod
    def _logp_from_scores(scores: List[torch.Tensor], seqs: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities from generation scores"""
        lp = torch.stack([F.log_softmax(s.float(), -1) for s in scores], dim=1)
        gen_tok = seqs[:, -lp.size(1):]
        return lp.gather(2, gen_tok.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate text using the LLM"""
        device = next(self.model.parameters()).device
        tok_in = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        tok_in = {k: v.to(device) for k, v in tok_in.items()}
        
        out = self.model.generate(
            **tok_in,
            max_new_tokens=128,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            renormalize_logits=True
        )
        
        seqs = out.sequences
        lp_tok = self._logp_from_scores(out.scores, seqs)
        logp = lp_tok.sum(-1)
        texts = self.tokenizer.batch_decode(
            seqs[:, tok_in['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return {'sequences': seqs, 'texts': texts, 'logprobs': logp}

# ══════════════════ 6. Builder Functions ══════════════════
def build_enhanced_actor_and_critic(
    cfg: SHPPOModelConfig,
    roles: Dict[str, RoleConfig],
    device: Union[str, torch.device] = 'cpu'
) -> Tuple[Dict[str, MultiAgentActor], MultiHeadCritic, ScalableInferenceNet]:
    """
    Build enhanced actors, critic, and scalable inference network
    
    Returns:
        actors: Dict of role -> MultiAgentActor
        critic: MultiHeadCritic
        inference_net: ScalableInferenceNet
    """
    device = torch.device(device)
    
    # Build actors for each role
    actors = {
        name: MultiAgentActor(rcfg, cfg).to(device)
        for name, rcfg in roles.items()
    }
    
    # Build critic
    critic = MultiHeadCritic(cfg.hete_layer_input_dim, list(roles.keys())).to(device)
    
    # Build scalable inference network
    inference_net = ScalableInferenceNet(cfg).to(device)
    
    return actors, critic, inference_net

# Legacy compatibility
def build_actor_and_critic(
    cfg: SHPPOModelConfig,
    roles: Dict[str, RoleConfig], 
    device: Union[str, torch.device] = 'cpu'
) -> Tuple[Dict[str, MultiAgentActor], MultiHeadCritic, ScalableInferenceNet]:
    """Legacy function for compatibility"""
    return build_enhanced_actor_and_critic(cfg, roles, device)

# Alias for InferenceNet
InferenceNet = ScalableInferenceNet