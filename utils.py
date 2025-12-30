"""
Utility functions for training and evaluation
"""
import torch
import torch.nn as nn
import numpy as np
import random
import os
from pathlib import Path

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_causal_mask(seq_len: int, device: str = "cpu") -> torch.Tensor:
    """
    Create causal mask for autoregressive language modeling

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask

def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create padding mask

    Args:
        seq: Input sequence (batch_size, seq_len)
        pad_idx: Padding token index

    Returns:
        Padding mask (batch_size, 1, 1, seq_len)
    """
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    path: str
):
    """Save model checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: str = "cpu"
):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded from {path}")
    print(f"Resuming from epoch {epoch}, step {step}, loss {loss:.4f}")

    return epoch, step, loss

class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
