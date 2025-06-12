from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from typing import List, Any, Dict, Union


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shppo.trainer import SHPPOConfig

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


# ══════════════════ LatentNet: Stochastic Encoder (Eq.4) ══════════════════
class LatentNet(nn.Module):
    """
    LatentNet – stochastic encoder producing a Gaussian latent vector *z*
    for each agent as described in the SH‑PPO paper (Eq. 4).

    • Input  : concatenation of observation embedding ϕ(o_t) and previous
               GRU hidden state h_{t‑1}  — dimension = cfg.hete_layer_input_dim * 2
    • Output : sampled latent z_t  ~ 𝒩(μ_t, σ_t),  along with μ_t and σ_t
    """
    def __init__(self, cfg):
        super().__init__()
        in_dim      = cfg.hete_layer_input_dim * 2
        hidden_dim  = cfg.mlp_hidden_dim
        latent_dim  = cfg.latent_dim

        # Two‑layer MLP encoder (paper Fig.2a)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Separate heads for μ and log σ (log std‑dev)
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_log = nn.Linear(hidden_dim, latent_dim)

        # Orthogonal init with small std as in paper
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 0.01 if m in (self.fc_mu, self.fc_log) else math.sqrt(2)
                nn.init.orthogonal_(m.weight, gain)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs_emb: torch.Tensor,      # (B, D)
        h_prev:  torch.Tensor,      # (B, D)
        sample:  bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z   : (B, latent_dim)  – sampled latent (or μ if sample=False)
        mu  : (B, latent_dim)
        sigma: (B, latent_dim)  – std‑dev (positive)
        """
        x = torch.cat([obs_emb, h_prev], dim=-1)           # (B, 2D)
        h = self.encoder(x)                                # (B, hidden)
        mu     = self.fc_mu(h)                             # (B, latent)
        log_sd = self.fc_log(h).clamp(-10, 10)             # avoid extremes
        sigma  = torch.exp(log_sd)

        if sample:
            eps = torch.randn_like(mu)
            z = mu + eps * sigma                           # reparameterize
        else:
            z = mu
        return z, mu, sigma


# ══════════════════ 3. Shared QLoRA LLM ══════════════════
class SharedLLM(nn.Module):
    def _logp_from_scores(self, prompts: List[str], sequences: torch.LongTensor) -> torch.Tensor:
        """
        Recompute log-probs of `sequences` under the current LoRA policy given `prompts`.
        Returns tensor of shape (B,) with sum of token log-probs.
        """
        # Tokenize prompts
        tok = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = tok['input_ids'].to(self.model.device)
        attention_mask = tok['attention_mask'].to(self.model.device)

        # Concatenate generation tokens
        gen_ids = sequences[:, input_ids.size(1):]
        # Build full input
        full_ids = torch.cat([input_ids, gen_ids], dim=1)
        full_mask = torch.cat([attention_mask, torch.ones_like(gen_ids)], dim=1)

        # Forward once to get logits
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_mask,
            return_dict=True
        )
        logits = outputs.logits  # (B, L, V)
        # Compute log-probs of generation tokens
        # Shift logits to align with next token
        shift_logits = logits[:, :-1, :].float()
        shift_labels = full_ids[:, 1:]
        logp_all = F.log_softmax(shift_logits, dim=-1)
        # Gather only gen positions
        gen_logp = logp_all[:, input_ids.size(1)-1:-1, :].gather(
            2, gen_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, gen_len)
        return gen_logp.sum(dim=1)
    """
    Wraps a 4-bit QLoRA model for one agent role.
    """
    def __init__(self, cfg: SHPPOConfig, device: Union[str, torch.device]):
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
        for _, param in self.model.named_parameters():
            param.requires_grad = ('lora_' in _ or 'Lora' in _)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.cfg = cfg

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> Dict[str, Any]:
        device = next(self.model.parameters()).device
        tok = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        tok = {k: v.to(device) for k, v in tok.items()}
        out = self.model.generate(
            **tok,
            max_new_tokens=self.cfg.max_solution_length or self.cfg.hete_layer_input_dim,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            renormalize_logits=True
        )
        seqs = out.sequences
        # compute log-probs
        lp = torch.stack([F.log_softmax(s.float(), -1) for s in out.scores], dim=1)
        gen_tok = seqs[:, tok['input_ids'].shape[1]:]
        logp = lp.gather(2, gen_tok.unsqueeze(-1)).squeeze(-1).sum(-1)
        texts = self.tokenizer.batch_decode(gen_tok, skip_special_tokens=True)
        return {'sequences': seqs, 'texts': texts, 'logprobs': logp}

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]


# ══════════════════ MultiHeadCritic ══════════════════
class MultiHeadCritic(nn.Module):
    """
    Simple critic head that maps latent features to a scalar value.
    """
    def __init__(self, in_dim: int, role_names: List[str]):
        super().__init__()
        # single shared value head
        self.fc = nn.Linear(in_dim, 1)
        # orthogonal init for stability
        nn.init.orthogonal_(self.fc.weight, math.sqrt(2))
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) -> returns (B, 1)
        return self.fc(x)

# ══════════════════ Heterogeneous Decoder & Layer ══════════════════
class HeteroDecoder(nn.Module):
    """
    MLP that maps latent z to weight & bias for a single hetero linear layer.
    """
    def __init__(self, latent_dim: int, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, (in_dim + 1) * out_dim),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          W: (B, out_dim, in_dim)
          b: (B, out_dim)
        """
        params = self.mlp(z)
        W, b = params.split([self.out_dim * self.in_dim, self.out_dim], dim=-1)
        W = W.view(-1, self.out_dim, self.in_dim)
        return W, b

def hetero_linear(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Applies a batch of linear transforms: x @ W^T + b
    x: (B, in_dim)
    W: (B, out_dim, in_dim)
    b: (B, out_dim)
    returns: (B, out_dim)
    """
    return torch.baddbmm(b.unsqueeze(1), W, x.unsqueeze(-1)).squeeze(-1)

# ══════════════════ SHPPOActor: Latent + HeteroDecoder + SharedLLM ══════════════════
class SHPPOActor(nn.Module):
    """
    Actor wrapper: applies LatentNet + HeteroDecoder to modulate
    the SharedLLM per-sample adapter weights before generation.
    """
    def __init__(self, cfg: SHPPOConfig, device: Union[str, torch.device]):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.latent_net = LatentNet(cfg).to(self.device)
        self.decoder    = HeteroDecoder(cfg.latent_dim,
                                        cfg.hete_layer_input_dim,
                                        cfg.hete_layer_output_dim).to(self.device)
        self.llm        = SharedLLM(cfg, self.device)
        # Cache original LoRA adapter weights for clean injection
        self._base_lora_weights: Dict[str, torch.Tensor] = {}
        self._base_lora_biases:  Dict[str, torch.Tensor] = {}
        for name, module in self.llm.model.named_modules():
            if any(tm in name for tm in self.cfg.target_modules) and hasattr(module, "weight"):
                self._base_lora_weights[name] = module.weight.data.clone()
                # Already cached weight; also cache bias if present
                if hasattr(module, "lora_A") and hasattr(module.lora_A, "bias"):
                    self._base_lora_biases[name] = module.lora_A.bias.data.clone()
                else:
                    self._base_lora_biases[name] = None

    def generate(self, prompts: List[str], h_prev: Any = None) -> Dict[str, Any]:
        """
        1) Tokenize prompts, run forward pass to get transformer last hidden state as obs_emb.
        2) Pass obs_emb and h_prev into latent_net.
        3) Decode hetero weights W, inject into LoRA adapter (lora_A) for each sample.
        4) Call generate() after injection, return h_next.
        5) Return output dict including h_next.
        """
        B = len(prompts)
        # Tokenize prompts (list of prompt structures or strings)
        device = self.device
        # If prompts are list of list of dict (chat format), flatten to strings
        if isinstance(prompts[0], list) and isinstance(prompts[0][0], dict):
            # Use tokenizer's chat template if available
            # Otherwise, join role-content pairs
            prompt_texts = []
            for p in prompts:
                # Try to use tokenizer's apply_chat_template if available
                if hasattr(self.llm.tokenizer, "apply_chat_template"):
                    prompt_texts.append(self.llm.tokenizer.apply_chat_template(p, tokenize=False))
                else:
                    # Fallback: join as role: content
                    prompt_texts.append("\n".join(f"{d['role']}: {d['content']}" for d in p))
        else:
            prompt_texts = prompts
        tok = self.llm.tokenizer(prompt_texts, return_tensors='pt', padding=True, truncation=True)
        tok = {k: v.to(device) for k, v in tok.items()}
        # Forward transformer to get last hidden state
        with torch.no_grad():
            outputs = self.llm.model(
                **tok, output_hidden_states=True, use_cache=False
            )
        last_hidden = outputs.hidden_states[-1][:, -1, :].float()  # (B, D)
        obs_emb = last_hidden
        # h_prev: if None, zeros
        if h_prev is None:
            h_prev = torch.zeros_like(obs_emb)
        # LatentNet
        z, mu, sigma = self.latent_net(obs_emb, h_prev, sample=True)
        # HeteroDecoder
        W, b = self.decoder(z)
        # Inject hetero weights into LoRA adapter lora_A
        for i in range(B):
            # Restore true base LoRA adapter weights
            for name, module in self.llm.model.named_modules():
                if name in self._base_lora_weights:
                    module.lora_A.weight.data.copy_(self._base_lora_weights[name])
                    if name in self._base_lora_biases and self._base_lora_biases[name] is not None:
                        module.lora_A.bias.data.copy_(self._base_lora_biases[name])
            # Then inject
            for name, module in self.llm.model.named_modules():
                if name in self._base_lora_weights and hasattr(module, "lora_A"):
                    module.lora_A.weight.data += W[i]
        # Generate after injection
        out = self.llm.generate(prompt_texts)
        # h_next: last_hidden (transformer last hidden state)
        h_next = last_hidden
        # Output
        out.update({
            "mu": mu,
            "sigma": sigma,
            "h_next": h_next
        })
        return out

class InferenceNet(nn.Module):
    def __init__(self, cfg: SHPPOConfig, global_dim: int):
        super().__init__()
        # input: flattened mu + sigma + global feature
        in_dim = 2 * cfg.latent_dim + global_dim
        self.net = nn.Sequential(
            MLP(in_dim, cfg.mlp_hidden_dim, cfg.mlp_hidden_dim),
            nn.Linear(cfg.mlp_hidden_dim, 1)
        )

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        B = mu.size(0)
        x = torch.cat([
            mu.view(B, -1),
            sigma.view(B, -1),
            global_feat.view(B, -1)
        ], dim=-1)
        return self.net(x).squeeze(-1)

# 2. build_actor_and_critic 함수 수정
def build_actor_and_critic(
    cfg: SHPPOConfig,
    device: Union[str, torch.device] = 'cpu'
) -> tuple[list, nn.Module, nn.Module, nn.Module]:
    device = torch.device(device)
    actors = [SHPPOActor(cfg, device) for _ in cfg.role_configs]
    critic = MultiHeadCritic(cfg.hete_layer_input_dim,
                             [rc.role_name for rc in cfg.role_configs]
    ).to(device)
    latent_net = LatentNet(cfg).to(device)
    # Pass global_dim=4 for easy/medium/hard/unknown one-hot
    inference_net = InferenceNet(cfg, global_dim=4).to(device)
    return actors, critic, latent_net, inference_net