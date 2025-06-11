"""
model.py ― LoRA-adapted actor and scalar-value critic for PPO code agents
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig


# ════════════════════════════════════════════════════════════════════════════
# 1.  Actor
# ════════════════════════════════════════════════════════════════════════════
class LLMActor(nn.Module):
    """
    A lightweight wrapper around a 4-bit-quantised causal LLM with LoRA adapters.
    Exposes `generate()` that returns sequences, decoded texts and summed
    log-probs, all needed by the PPO trainer.
    """

    def __init__(self, cfg, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # 4-bit quantisation for memory efficiency
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            device_map="auto",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )

        # LoRA adapters
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)

        # Freeze all base weights
        for n, p in self.model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model_name, use_fast=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _logprobs_from_scores(self, scores, seqs):
        """
        Convert `generate(..., output_scores=True)` outputs into per-token
        log-probs for the generated suffix.
        """
        # scores: list[T_gen] of [B, vocab]
        logps = torch.stack(
            [F.log_softmax(s.float(), dim=-1) for s in scores], dim=1
        )                              # [B, T_gen, vocab]
        gen_tokens = seqs[:, -logps.size(1):]         # strip prompt
        token_logps = logps.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)
        return token_logps

    # ------------------------------------------------------------------ #
    # Public API used by the trainer
    # ------------------------------------------------------------------ #
    def generate(self, prompts: list[str]) -> dict[str, torch.Tensor | list[str]]:
        """
        Returns:
            sequences  – LongTensor[B, T_total]
            texts      – list[str] decoded solutions
            logprobs   – FloatTensor[B]  (sum over generated tokens)
        """
        
        # Tokenize inputs on the model's device
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_problem_length,
        )
        
        embed_device = next(self.model.base_model.model.model.embed_tokens.parameters()).device
        inputs = {k: v.to(embed_device) for k, v in inputs.items()}

        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_solution_length,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            renormalize_logits=True,
            remove_invalid_values=True,
        )

        seqs = gen_out.sequences              # [B, T_prompt + T_gen]
        logps = self._logprobs_from_scores(gen_out.scores, seqs).sum(dim=-1)

        texts = self.tokenizer.batch_decode(
            seqs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return {"sequences": seqs, "texts": texts, "logprobs": logps}

    # Simple passthrough for trainer’s hidden-state extraction
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # Convenience
    def get_trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]


# ════════════════════════════════════════════════════════════════════════════
# 2.  Critic
# ════════════════════════════════════════════════════════════════════════════
class Critic(nn.Module):
    """
    Scalar value head for PPO.

    * If `cfg.critic_hidden_dims` is empty ➞ single linear layer.
    * Otherwise ➞ small MLP ending in tanh to bound outputs to (-1, 1).
    """

    def __init__(self, hidden_dim: int, cfg):
        super().__init__()
        if not cfg.critic_hidden_dims:
            self.net = nn.Linear(hidden_dim, 1)
        else:
            layers = []
            inp = hidden_dim
            for h in cfg.critic_hidden_dims:
                layers += [nn.Linear(inp, h), nn.ReLU()]
                inp = h
            layers += [nn.Linear(inp, 1), nn.Tanh()]
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def get_trainable_parameters(self):
        return self.parameters()


# ════════════════════════════════════════════════════════════════════════════
# 3.  Factory
# ════════════════════════════════════════════════════════════════════════════
def build_actor_and_critic(cfg, device):
    """
    Convenience wrapper used by `ppo.trainer.PPOTrainer`.
    """
    actor = LLMActor(cfg, device)
    hidden_size = actor.model.config.hidden_size
    critic = Critic(hidden_size, cfg).to(device)
    # print trainable parameters
    actor_params = sum(p.numel() for p in actor.get_trainable_parameters() if p.requires_grad)
    critic_params = sum(p.numel() for p in critic.get_trainable_parameters() if p.requires_grad)
    print(f"Tainable parameters: Actor={actor_params:,}, Critic={critic_params:,}, Total={actor_params + critic_params:,}")
    return (
        actor,
        critic,
        actor.get_trainable_parameters(),
        critic.get_trainable_parameters(),
        actor.tokenizer,
    )