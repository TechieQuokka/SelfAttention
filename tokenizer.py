"""
Tokenizer for text processing
"""
from typing import List, Dict
from pathlib import Path
import json
from collections import Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

class SimpleTokenizer:
    """Simple BPE-based tokenizer using HuggingFace tokenizers library"""

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def train(self, texts: List[str], save_path: str = None):
        """
        Train BPE tokenizer on texts

        Args:
            texts: List of text strings to train on
            save_path: Path to save trained tokenizer
        """
        # Initialize BPE tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token=self.unk_token))

        # Configure pre-tokenization (split on whitespace and punctuation)
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Configure decoder
        self.tokenizer.decoder = decoders.ByteLevel()

        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[self.pad_token, self.unk_token, self.bos_token, self.eos_token],
            show_progress=True
        )

        # Train tokenizer
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        # Configure post-processing to add special tokens
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[
                (self.bos_token, self.tokenizer.token_to_id(self.bos_token)),
                (self.eos_token, self.tokenizer.token_to_id(self.eos_token)),
            ],
        )

        # Enable padding
        self.tokenizer.enable_padding(
            pad_id=self.tokenizer.token_to_id(self.pad_token),
            pad_token=self.pad_token
        )

        # Enable truncation
        self.tokenizer.enable_truncation(max_length=512)

        if save_path:
            self.save(save_path)

        print(f"Tokenizer trained with vocabulary size: {self.tokenizer.get_vocab_size()}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")

        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encode batch of texts"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")

        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def save(self, path: str):
        """Save tokenizer to file"""
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from file"""
        self.tokenizer = Tokenizer.from_file(path)
        print(f"Tokenizer loaded from {path}")

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID"""
        return self.tokenizer.token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID"""
        return self.tokenizer.token_to_id(self.unk_token)

    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID"""
        return self.tokenizer.token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID"""
        return self.tokenizer.token_to_id(self.eos_token)

    def __len__(self):
        """Get vocabulary size"""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()


def train_tokenizer_from_dataset(dataset, vocab_size: int = 30000, save_path: str = "tokenizer/tokenizer.json"):
    """
    Train tokenizer from WikiText-2 dataset

    Args:
        dataset: HuggingFace dataset
        vocab_size: Vocabulary size
        save_path: Path to save tokenizer

    Returns:
        Trained tokenizer
    """
    print("Training tokenizer...")

    # Extract non-empty texts from training set
    texts = [sample['text'] for sample in dataset['train'] if sample['text'].strip()]

    # Initialize and train tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, save_path=save_path)

    # Test tokenizer
    test_text = "This is a test sentence for the tokenizer."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\nTest encoding:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    return tokenizer
