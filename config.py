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
    d_model: int = 256  # Embedding dimension
    n_heads: int = 4  # Number of attention heads
    n_layers: int = 3  # Number of transformer layers
    d_ff: int = 768  # Feed-forward dimension
    dropout: float = 0.35  # 0.4 → 0.35 (약간 완화)
    max_seq_len: int = 512  # Maximum sequence length

    # Training
    batch_size: int = 16
    learning_rate: float = 4e-5  # 3e-5 → 4e-5 (조금 높임)
    num_epochs: int = 50
    warmup_steps: int = 3000  # 2000 → 3000 (좀 더 안정적으로)
    gradient_clip: float = 0.5  # 1.0 → 0.5 (gradient 더 제한)
    weight_decay: float = 0.08  # 0.1 → 0.08 (약간 완화)

    # Data
    data_path: str = "data/wikitext2"
    tokenizer_path: str = "tokenizer"
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cuda"  # or "cpu"

    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 5000

    def __post_init__(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.vocab_size > 0, "vocab_size must be positive"

@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    seed: int = 16384
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True  # Use automatic mixed precision
    gradient_accumulation_steps: int = 4

    # Early stopping
    patience: int = 5
    min_delta: float = 0.0005  # 0.001 → 0.0005 (더 민감하게)

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine" or "linear"
    min_lr: float = 1e-6  # 5e-6 → 1e-6 (더 낮게, 더 오래 학습)