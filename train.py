"""
Training script for Transformer Language Model
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler  # 수정된 부분
from pathlib import Path
from tqdm import tqdm
import math
import os

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

        # Mixed precision training - 수정된 부분
        self.scaler = GradScaler('cuda') if train_config.mixed_precision else None

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

            # Forward pass with mixed precision - 수정된 부분
            if self.scaler is not None:
                with autocast('cuda'):  # device_type 인자 제거
                    logits = self.model(input_ids, attention_mask=padding_mask)
                    loss = self._compute_loss(logits, target_ids, padding_mask)
                    # Gradient accumulation 적용
                    loss = loss / self.train_config.gradient_accumulation_steps
            else:
                logits = self.model(input_ids, attention_mask=padding_mask)
                loss = self._compute_loss(logits, target_ids, padding_mask)
                loss = loss / self.train_config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation step마다 업데이트
                if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                loss.backward()
                
                if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

            # Update metrics (매 배치마다)
            losses.update(loss.item() * self.train_config.gradient_accumulation_steps, input_ids.size(0))
            
            # 메모리 해제
            del logits, loss, padding_mask
            if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                torch.cuda.empty_cache()
            
            # Progress bar는 매 배치마다 업데이트
            progress_bar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'lr': f'{get_lr(self.optimizer):.2e}',
                'ppl': f'{math.exp(min(losses.avg, 10)):.2f}'  # overflow 방지
            })
            
            # step은 실제 optimizer step마다만 증가
            if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                self.global_step += 1

                # Log and evaluate (optimizer step마다)
                if self.global_step % self.config.log_interval == 0:
                    print(f"\nStep {self.global_step} - Train Loss: {losses.avg:.4f}, "
                        f"Perplexity: {math.exp(min(losses.avg, 10)):.2f}")

                if self.global_step % self.config.eval_interval == 0:
                    val_loss = self.validate()
                    print(f"Step {self.global_step} - Val Loss: {val_loss:.4f}, "
                        f"Perplexity: {math.exp(min(val_loss, 10)):.2f}")

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
            
            # 메모리 해제
            del input_ids, target_ids, logits, loss, padding_mask

        torch.cuda.empty_cache()
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

            # Compute loss with label smoothing
            loss = nn.functional.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=self.model.pad_idx,
                label_smoothing=0.1,
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


import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """Cosine schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup: 0에서 1까지 선형 증가
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay: 1에서 min_lr_ratio까지 감소
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def main():
    """Main training function"""

    # Tokenizers parallelism 경고 제거
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        pin_memory=False  # pin_memory False로 변경
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
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * config.num_epochs // train_config.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=train_config.min_lr / config.learning_rate
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