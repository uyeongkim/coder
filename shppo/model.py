from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union

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
        latent = mu + sig * torch.randn_like(sig)
        return latent, mu, sig

class TemplateDecoder(nn.Module):
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

# ══════════════════ 3. Shared 4-bit QLoRA LLM ══════════════════
class SharedLLM(nn.Module):
    def __init__(self, cfg: SHPPOModelConfig, device: Union[str, torch.device]):
        super().__init__()
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            device_map='auto',
            quantization_config=bnb,
            trust_remote_code=True
        )
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            bias='none',
            task_type='CAUSAL_LM'
        )
        self.model = get_peft_model(base, lora_cfg).to(device).eval()
        for name, param in self.model.named_parameters():
            if 'lora_' in name or 'Lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        # 디버깅: LoRA 파라미터 확인
        lora_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        # print(f"LoRA parameters: {lora_params}")
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name,
                                                      trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        cfg.dtype = self.model.dtype
        self.cfg = cfg

    @staticmethod
    def _logp_from_scores(scores: List[torch.Tensor], seqs: torch.Tensor) -> torch.Tensor:
        lp = torch.stack([F.log_softmax(s.float(), -1) for s in scores], dim=1)
        gen_tok = seqs[:, -lp.size(1):]
        return lp.gather(2, gen_tok.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> Dict[str, Any]:
        device = next(self.model.parameters()).device
        tok_in = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.cfg.hete_layer_input_dim
        )
        tok_in = {k: v.to(device) for k, v in tok_in.items()}
        
        out = self.model.generate(
            **tok_in,
            max_new_tokens=self.cfg.hete_layer_input_dim,
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

    def trainable_parameters(self) -> List[torch.Tensor]:
        """Return list of trainable LoRA parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

# ══════════════════ 4. Inference Network ══════════════════
# model.py에서 다음 부분들을 수정하세요:

# 1. InferenceNet 클래스 수정
class InferenceNet(nn.Module):
    def __init__(self, cfg: SHPPOModelConfig, max_n_templates: int = 8):
        super().__init__()
        # input: flattened mu + sigma + global hidden state
        in_dim = 2 * cfg.latent_dim * max_n_templates + cfg.hete_layer_input_dim
        self.net = nn.Sequential(
            MLP(in_dim, cfg.mlp_hidden_dim, cfg.mlp_hidden_dim),
            nn.Linear(cfg.mlp_hidden_dim, 1)
        )

    def forward(
        self,
        mu: torch.Tensor,
        sig: torch.Tensor,
        global_h: torch.Tensor
    ) -> torch.Tensor:
        B = mu.size(0)
        # Flatten and concatenate
        mu_flat = mu.reshape(B, -1)
        sig_flat = sig.reshape(B, -1)
        
        # Pad to consistent size if needed
        max_size = self.net[0].net[0].in_features - global_h.size(-1)
        current_size = mu_flat.size(-1) + sig_flat.size(-1)
        
        if current_size < max_size:
            padding = torch.zeros(B, max_size - current_size, device=mu.device)
            x = torch.cat([mu_flat, sig_flat, padding, global_h], dim=-1)
        else:
            # Truncate if too large
            combined = torch.cat([mu_flat, sig_flat], dim=-1)
            truncated = combined[:, :max_size]
            x = torch.cat([truncated, global_h], dim=-1)
        
        return self.net(x).squeeze(-1)

# 2. build_actor_and_critic 함수 수정
def build_actor_and_critic(
    cfg: SHPPOModelConfig,
    roles: Dict[str, RoleConfig],
    device: Union[str, torch.device] = 'cpu'
) -> tuple[Dict[str, MultiAgentActor], MultiHeadCritic, InferenceNet]:
    """
    Returns actors, critic, and inference network.
    """
    device = torch.device(device)
    actors = {
        name: MultiAgentActor(rcfg, cfg).to(device)
        for name, rcfg in roles.items()
    }
    critic = MultiHeadCritic(cfg.hete_layer_input_dim, list(roles.keys())).to(device)
    
    # Calculate max templates across all roles
    max_templates = max(rcfg.n_action_templates for rcfg in roles.values())
    inference_net = InferenceNet(cfg, max_templates).to(device)
    
    return actors, critic, inference_net