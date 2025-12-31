"""
Transformer LLM Configuration
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Transformer 모델 설정"""
    # Model architecture - Phase 1 개선
    vocab_size: int = 30000
    d_model: int = 512  # 256 → 512 (2배 증가)
    n_heads: int = 8    # 4 → 8 (2배 증가)
    n_layers: int = 6   # 3 → 6 (2배 증가)
    d_ff: int = 2048    # 768 → 2048 (2.7배 증가)
    dropout: float = 0.1  # 0.35 → 0.1 (정규화 완화)
    max_seq_len: int = 512  # Maximum sequence length

    # Training - Phase 1 개선
    batch_size: int = 16  # GPU 메모리에 따라 조정 가능
    learning_rate: float = 2e-4  # 4e-5 → 2e-4 (5배 증가)
    num_epochs: int = 50
    warmup_steps: int = 4000  # 3000 → 4000 (모델 커져서 증가)
    gradient_clip: float = 1.0  # 0.5 → 1.0 (완화)
    weight_decay: float = 0.01  # 0.08 → 0.01 (정규화 완화)

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

    # Regularization
    label_smoothing: float = 0.1  # Applied during training only (not validation/test)

    # Early stopping
    patience: int = 5
    min_delta: float = 0.0005  # 0.001 → 0.0005 (더 민감하게)

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine" or "linear"
    min_lr: float = 1e-6  # 5e-6 → 1e-6 (더 낮게, 더 오래 학습)