"""
Training script for Transformer Language Model
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import math

from model import TransformerLM
from dataset import prepare_dataloaders
from tokenizer import SimpleTokenizer, train_tokenizer_from_dataset
from config import ModelConfig, TrainingConfig
from utils import (
    set_seed, count_parameters, save_checkpoint,
    load_checkpoint, AverageMeter, get_lr
)
from datasets import load_from_disk

class Trainer:
    """Trainer for Transformer Language Model"""

    def __init__(
        self,
        model: TransformerLM,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config: ModelConfig,
        train_config: TrainingConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.train_config = train_config
        self.device = device

        # Mixed precision training
        self.scaler = GradScaler() if train_config.mixed_precision else None

        # Metrics
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        losses = AverageMeter()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Create padding mask
            padding_mask = (input_ids == self.model.pad_idx)

            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    logits = self.model(input_ids, attention_mask=padding_mask)
                    loss = self._compute_loss(logits, target_ids, padding_mask)
            else:
                logits = self.model(input_ids, attention_mask=padding_mask)
                loss = self._compute_loss(logits, target_ids, padding_mask)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            self.scheduler.step()

            # Update metrics
            losses.update(loss.item(), input_ids.size(0))
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'lr': f'{get_lr(self.optimizer):.2e}',
                'ppl': f'{math.exp(losses.avg):.2f}'
            })

            # Log and evaluate
            if self.global_step % self.config.log_interval == 0:
                print(f"\nStep {self.global_step} - Train Loss: {losses.avg:.4f}, "
                      f"Perplexity: {math.exp(losses.avg):.2f}")

            if self.global_step % self.config.eval_interval == 0:
                val_loss = self.validate()
                print(f"Step {self.global_step} - Val Loss: {val_loss:.4f}, "
                      f"Perplexity: {math.exp(val_loss):.2f}")

                # Check for improvement
                if val_loss < self.best_val_loss - self.train_config.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_best_checkpoint(epoch, val_loss)
                else:
                    self.patience_counter += 1

                self.model.train()

            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(epoch, losses.avg)

        return losses.avg

    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        losses = AverageMeter()

        for input_ids, target_ids in tqdm(self.val_loader, desc="Validating"):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            padding_mask = (input_ids == self.model.pad_idx)

            logits = self.model(input_ids, attention_mask=padding_mask)
            loss = self._compute_loss(logits, target_ids, padding_mask)

            losses.update(loss.item(), input_ids.size(0))

        return losses.avg

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss ignoring padding tokens"""
        # Flatten for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.model.pad_idx,
            reduction='mean'
        )

        return loss

    def _save_checkpoint(self, epoch: int, loss: float):
        """Save regular checkpoint"""
        save_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.global_step}.pt"
        save_checkpoint(self.model, self.optimizer, epoch, self.global_step, loss, str(save_path))

    def _save_best_checkpoint(self, epoch: int, loss: float):
        """Save best checkpoint"""
        save_path = Path(self.config.checkpoint_dir) / "best_model.pt"
        save_checkpoint(self.model, self.optimizer, epoch, self.global_step, loss, str(save_path))
        print(f"New best model saved with val loss: {loss:.4f}")

    def train(self):
        """Full training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Total steps per epoch: {len(self.train_loader)}")

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"{'='*50}")

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_loss:.4f}, Perplexity: {math.exp(train_loss):.2f}")
            print(f"Val Loss: {val_loss:.4f}, Perplexity: {math.exp(val_loss):.2f}")

            # Early stopping
            if self.patience_counter >= self.train_config.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main training function"""
    # Configuration
    config = ModelConfig()
    train_config = TrainingConfig()

    # Set seed
    set_seed(train_config.seed)

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(config.data_path)

    # Train or load tokenizer
    tokenizer_path = Path(config.tokenizer_path) / "tokenizer.json"
    if tokenizer_path.exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        tokenizer.load(str(tokenizer_path))
    else:
        print("Training new tokenizer...")
        tokenizer = train_tokenizer_from_dataset(
            dataset,
            vocab_size=config.vocab_size,
            save_path=str(tokenizer_path)
        )

    # Update vocab size in config
    config.vocab_size = len(tokenizer)
    print(f"Vocabulary size: {config.vocab_size}")

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        config.data_path,
        tokenizer,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )

    # Create model
    print("\nInitializing model...")
    model = TransformerLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        pad_idx=tokenizer.pad_token_id
    ).to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=train_config.min_lr
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        train_config=train_config,
        device=device
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
