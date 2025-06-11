"""
model.py - Model Management
Actor, Critic, and model loading utilities
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Tuple, List
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig

import logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    
    # LoRA settings
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Quantization settings
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_use_double_quant: bool = True
    llm_int8_enable_fp32_cpu_offload: bool = True
    
    # Training settings
    torch_dtype: torch.dtype = torch.bfloat16
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = True


class Critic(nn.Module):
    """Critic network for value function estimation"""
    
    def __init__(self, hidden_dim: int, hidden_layers: int = 3):
        super().__init__()
        
        layers = []
        current_dim = hidden_dim
        
        for i in range(hidden_layers):
            next_dim = max(128, current_dim // 2)
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.net(x).squeeze(-1)  # (batch_size,)


class ModelManager:
    """Model loading and management"""
    
    @staticmethod
    def load_tokenizer(config: ModelConfig) -> AutoTokenizer:
        """Load tokenizer"""
        logger.info(f"Loading tokenizer: {config.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.padding_side = "left"
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    
    @staticmethod
    def load_base_model(config: ModelConfig) -> AutoModelForCausalLM:
        """Load base model with quantization"""
        logger.info(f"Loading base model: {config.model_name}")
        
        # Quantization configuration
        bnb_config = None
        if config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                llm_int8_enable_fp32_cpu_offload=config.llm_int8_enable_fp32_cpu_offload
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=config.torch_dtype,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
            trust_remote_code=config.trust_remote_code
        )
        
        # Disable sliding window attention if exists
        if hasattr(model.config, 'use_sliding_window_attention'):
            model.config.use_sliding_window_attention = False
        
        logger.info("Base model loaded successfully")
        return model
    
    @staticmethod
    def apply_lora(model: AutoModelForCausalLM, config: ModelConfig) -> AutoModelForCausalLM:
        """Apply LoRA to model"""
        logger.info("Applying LoRA...")
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        lora_model = get_peft_model(model, lora_config)
        lora_model.train()
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())
        
        logger.info(f"LoRA applied successfully")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        return lora_model
    
    @staticmethod
    def create_critic(base_model: AutoModelForCausalLM, device: torch.device) -> Critic:
        """Create critic network"""
        logger.info("Creating critic network...")
        
        hidden_dim = base_model.config.hidden_size
        critic = Critic(hidden_dim).to(device)
        
        param_count = sum(p.numel() for p in critic.parameters())
        logger.info(f"Critic created with {param_count:,} parameters")
        
        return critic
    
    @staticmethod
    def get_trainable_parameters(actor: AutoModelForCausalLM, critic: Critic) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
        """Get trainable parameters for actor and critic"""
        # Actor: only LoRA parameters
        actor_params = [p for n, p in actor.named_parameters() if "lora_" in n and p.requires_grad]
        
        # Critic: all parameters
        critic_params = list(critic.parameters())
        
        logger.info(f"Actor trainable parameters: {len(actor_params)}")
        logger.info(f"Critic parameters: {len(critic_params)}")
        
        return actor_params, critic_params
    
    @staticmethod
    def load_full_model(config: ModelConfig, device: torch.device) -> Tuple[AutoModelForCausalLM, Critic, AutoTokenizer, List[torch.nn.Parameter], List[torch.nn.Parameter]]:
        """Load complete model setup"""
        logger.info("Loading complete model setup...")
        
        # Load tokenizer
        tokenizer = ModelManager.load_tokenizer(config)
        
        # Load base model
        base_model = ModelManager.load_base_model(config)
        
        # Apply LoRA
        actor = ModelManager.apply_lora(base_model, config)
        
        # Create critic
        critic = ModelManager.create_critic(base_model, device)
        
        # Get trainable parameters
        actor_params, critic_params = ModelManager.get_trainable_parameters(actor, critic)
        
        logger.info("Complete model setup loaded successfully")
        
        return actor, critic, tokenizer, actor_params, critic_params


class ModelUtils:
    """Model utility functions"""
    
    @staticmethod
    def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
        """Count model parameters"""
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def get_model_memory_usage(model: nn.Module) -> dict:
        """Get model memory usage information"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'parameter_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'total_size_mb': total_size / 1024 / 1024
        }
    
    @staticmethod
    def freeze_model(model: nn.Module):
        """Freeze all model parameters"""
        for param in model.parameters():
            param.requires_grad = False
    
    @staticmethod
    def unfreeze_model(model: nn.Module):
        """Unfreeze all model parameters"""
        for param in model.parameters():
            param.requires_grad = True
    
    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: List[str]):
        """Freeze specific layers"""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    @staticmethod
    def save_model_checkpoint(model: nn.Module, optimizer, step: int, loss: float, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model checkpoint saved to {filepath}")
    
    @staticmethod
    def load_model_checkpoint(model: nn.Module, optimizer, filepath: str, device: torch.device) -> Tuple[int, float]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        step = checkpoint['step']
        loss = checkpoint['loss']
        
        logger.info(f"Model checkpoint loaded from {filepath}")
        logger.info(f"Resumed from step {step} with loss {loss:.4f}")
        
        return step, loss
    
    @staticmethod
    def print_model_info(actor: nn.Module, critic: nn.Module):
        """Print comprehensive model information"""
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        
        # Actor information
        actor_trainable = ModelUtils.count_parameters(actor, trainable_only=True)
        actor_total = ModelUtils.count_parameters(actor, trainable_only=False)
        actor_memory = ModelUtils.get_model_memory_usage(actor)
        
        print(f"ACTOR (with LoRA):")
        print(f"  - Trainable parameters: {actor_trainable:,}")
        print(f"  - Total parameters: {actor_total:,}")
        print(f"  - Trainable ratio: {actor_trainable/actor_total*100:.2f}%")
        print(f"  - Memory usage: {actor_memory['total_size_mb']:.1f} MB")
        
        # Critic information
        critic_params = ModelUtils.count_parameters(critic, trainable_only=False)
        critic_memory = ModelUtils.get_model_memory_usage(critic)
        
        print(f"\nCRITIC:")
        print(f"  - Parameters: {critic_params:,}")
        print(f"  - Memory usage: {critic_memory['total_size_mb']:.1f} MB")
        
        # Total information
        total_trainable = actor_trainable + critic_params
        print(f"\nTOTAL TRAINABLE PARAMETERS: {total_trainable:,}")
        print("="*60)


class GenerationConfig:
    """Configuration for text generation"""
    
    def __init__(self):
        self.max_new_tokens = 512
        self.min_new_tokens = 20
        self.do_sample = True
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.repetition_penalty = 1.1
        self.length_penalty = 1.0
        self.use_cache = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'max_new_tokens': self.max_new_tokens,
            'min_new_tokens': self.min_new_tokens,
            'do_sample': self.do_sample,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'repetition_penalty': self.repetition_penalty,
            'length_penalty': self.length_penalty,
            'use_cache': self.use_cache
        }


def test_model_loading():
    """Test model loading functionality"""
    print("Testing model loading...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create configuration
    config = ModelConfig()
    print(f"Model config: {config.model_name}")
    
    # Test tokenizer loading
    tokenizer = ModelManager.load_tokenizer(config)
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")
    
    # Test critic creation (without full model for speed)
    hidden_dim = 4096  # typical hidden dimension
    critic = Critic(hidden_dim).to(device)
    critic_params = ModelUtils.count_parameters(critic)
    print(f"Critic created: {critic_params:,} parameters")
    
    print("Model loading test completed successfully!")


if __name__ == "__main__":
    test_model_loading()