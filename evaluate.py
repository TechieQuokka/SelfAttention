"""
Evaluation script for Transformer Language Model
"""
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import math

from model import TransformerLM
from tokenizer import SimpleTokenizer
from dataset import prepare_dataloaders
from config import ModelConfig
from utils import AverageMeter

@torch.no_grad()
def evaluate_model(model: TransformerLM, test_loader, device: str = "cuda"):
    """
    Evaluate model on test set

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    losses = AverageMeter()
    total_tokens = 0

    print("\nEvaluating model on test set...")

    for input_ids, target_ids in tqdm(test_loader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Create padding mask
        padding_mask = (input_ids == model.pad_idx)

        # Forward pass
        logits = model(input_ids, attention_mask=padding_mask)

        # Compute loss (no label smoothing for evaluation)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)

        loss = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=model.pad_idx,
            label_smoothing=0.0,  # No label smoothing for fair evaluation
            reduction='mean'
        )

        # Count non-padding tokens
        non_pad_tokens = (target_ids != model.pad_idx).sum().item()
        total_tokens += non_pad_tokens

        # Update metrics
        losses.update(loss.item(), input_ids.size(0))

    # Compute perplexity
    avg_loss = losses.avg
    perplexity = math.exp(avg_loss)

    metrics = {
        'test_loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens
    }

    return metrics


def print_metrics(metrics: dict):
    """Pretty print evaluation metrics"""
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Test Loss:     {metrics['test_loss']:.4f}")
    print(f"Perplexity:    {metrics['perplexity']:.2f}")
    print(f"Total Tokens:  {metrics['total_tokens']:,}")
    print("=" * 50)


def load_model_for_evaluation(checkpoint_path: str, config: ModelConfig, device: str = "cuda"):
    """Load trained model for evaluation"""
    # Load tokenizer
    tokenizer_path = Path(config.tokenizer_path) / "tokenizer.json"
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.load(str(tokenizer_path))

    # Create model
    model = TransformerLM(
        vocab_size=len(tokenizer),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        pad_idx=tokenizer.pad_token_id
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}, "
          f"Training Loss: {checkpoint['loss']:.4f}")

    return model, tokenizer


def main():
    """Main evaluation function"""
    config = ModelConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load model
    checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return

    model, tokenizer = load_model_for_evaluation(str(checkpoint_path), config, device)

    # Prepare test dataloader
    print("\nPreparing test data...")
    _, _, test_loader = prepare_dataloaders(
        config.data_path,
        tokenizer,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)

    # Print results
    print_metrics(metrics)

    # Save results
    results_path = Path(config.checkpoint_dir) / "evaluation_results.txt"
    with open(results_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Loss:     {metrics['test_loss']:.4f}\n")
        f.write(f"Perplexity:    {metrics['perplexity']:.2f}\n")
        f.write(f"Total Tokens:  {metrics['total_tokens']:,}\n")
        f.write("=" * 50 + "\n")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
