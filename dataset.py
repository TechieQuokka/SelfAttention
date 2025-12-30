"""
Dataset preparation for language modeling
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from datasets import load_from_disk
from tokenizer import SimpleTokenizer

class LanguageModelingDataset(Dataset):
    """Dataset for autoregressive language modeling"""

    def __init__(
        self,
        texts: List[str],
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 512,
        stride: int = 256
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            stride: Stride for creating overlapping sequences
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride

        # Tokenize all texts and create sequences
        self.sequences = self._create_sequences(texts)

        print(f"Created {len(self.sequences)} sequences from {len(texts)} texts")

    def _create_sequences(self, texts: List[str]) -> List[List[int]]:
        """Create training sequences from texts"""
        all_sequences = []

        for text in texts:
            if not text.strip():
                continue

            # Encode text
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            # Skip if too short
            if len(token_ids) < 2:
                continue

            # Create sequences with sliding window
            for i in range(0, len(token_ids) - 1, self.stride):
                seq = token_ids[i:i + self.max_seq_len]
                if len(seq) >= 2:  # Need at least 2 tokens for input-target pair
                    all_sequences.append(seq)

        return all_sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input-target pair for language modeling

        Returns:
            input_ids: Input sequence (all tokens except last)
            target_ids: Target sequence (all tokens except first)
        """
        seq = self.sequences[idx]

        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)

        return input_ids, target_ids


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int = 0):
    """
    Collate function for DataLoader with padding

    Args:
        batch: List of (input_ids, target_ids) tuples
        pad_idx: Padding index

    Returns:
        Padded batch of inputs and targets
    """
    inputs, targets = zip(*batch)

    # Get max length in batch
    max_len = max(len(seq) for seq in inputs)

    # Pad sequences
    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        padded_inputs.append(
            torch.cat([inp, torch.full((pad_len,), pad_idx, dtype=torch.long)])
        )
        padded_targets.append(
            torch.cat([tgt, torch.full((pad_len,), pad_idx, dtype=torch.long)])
        )

    return torch.stack(padded_inputs), torch.stack(padded_targets)


def prepare_dataloaders(
    dataset_path: str,
    tokenizer: SimpleTokenizer,
    batch_size: int = 32,
    max_seq_len: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare train, validation, and test dataloaders

    Args:
        dataset_path: Path to WikiText-2 dataset
        tokenizer: Trained tokenizer
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory

    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)

    # Extract non-empty texts
    train_texts = [s['text'] for s in dataset['train'] if s['text'].strip()]
    val_texts = [s['text'] for s in dataset['validation'] if s['text'].strip()]
    test_texts = [s['text'] for s in dataset['test'] if s['text'].strip()]

    print(f"Train texts: {len(train_texts)}")
    print(f"Validation texts: {len(val_texts)}")
    print(f"Test texts: {len(test_texts)}")

    # Create datasets
    train_dataset = LanguageModelingDataset(
        train_texts, tokenizer, max_seq_len=max_seq_len, stride=max_seq_len // 2
    )
    val_dataset = LanguageModelingDataset(
        val_texts, tokenizer, max_seq_len=max_seq_len, stride=max_seq_len
    )
    test_dataset = LanguageModelingDataset(
        test_texts, tokenizer, max_seq_len=max_seq_len, stride=max_seq_len
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda b: collate_fn(b, pad_idx=tokenizer.pad_token_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda b: collate_fn(b, pad_idx=tokenizer.pad_token_id)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda b: collate_fn(b, pad_idx=tokenizer.pad_token_id)
    )

    return train_loader, val_loader, test_loader
