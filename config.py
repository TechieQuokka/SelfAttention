"""
Transformer LLM Configuration
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Transformer 모델 설정"""
    # Model architecture
    vocab_size: int = 30000
    d_model: int = 512  # Embedding dimension
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 6  # Number of transformer layers
    d_ff: int = 2048  # Feed-forward dimension
    dropout: float = 0.1
    max_seq_len: int = 512  # Maximum sequence length

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 10
    warmup_steps: int = 4000
    gradient_clip: float = 1.0

    # Data
    data_path: str = "data/wikitext2"
    tokenizer_path: str = "tokenizer"
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cuda"  # or "cpu"

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    def __post_init__(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.vocab_size > 0, "vocab_size must be positive"

@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True  # Use automatic mixed precision
    gradient_accumulation_steps: int = 1

    # Early stopping
    patience: int = 3
    min_delta: float = 0.001

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine" or "linear"
    min_lr: float = 1e-6
