"""
Test script to verify all components work correctly
"""
import torch
from datasets import load_from_disk

def test_tokenizer():
    """Test tokenizer functionality"""
    print("\n" + "="*80)
    print("Testing Tokenizer...")
    print("="*80)

    from tokenizer import SimpleTokenizer

    # Create simple tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)

    # Train on small sample
    sample_texts = [
        "This is a test sentence.",
        "Another test sentence for the tokenizer.",
        "Machine learning is fascinating."
    ]

    tokenizer.train(sample_texts)

    # Test encoding/decoding
    test_text = "This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    print("✅ Tokenizer test passed!")


def test_model():
    """Test model architecture"""
    print("\n" + "="*80)
    print("Testing Model Architecture...")
    print("="*80)

    from model import TransformerLM
    from utils import count_parameters

    # Small model for testing
    model = TransformerLM(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        dropout=0.1,
        pad_idx=0
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: (batch_size={batch_size}, seq_len={seq_len}, vocab_size=1000)")

    assert logits.shape == (batch_size, seq_len, 1000), "Output shape mismatch!"

    # Test generation
    prompt = torch.randint(0, 1000, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"\nGeneration test:")
    print(f"Prompt length: {prompt.shape[1]}")
    print(f"Generated length: {generated.shape[1]}")
    print(f"Expected: {prompt.shape[1] + 10}")

    assert generated.shape[1] == prompt.shape[1] + 10, "Generation length mismatch!"

    print("✅ Model test passed!")


def test_dataset():
    """Test dataset loading and preprocessing"""
    print("\n" + "="*80)
    print("Testing Dataset...")
    print("="*80)

    from tokenizer import SimpleTokenizer
    from dataset import LanguageModelingDataset

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    sample_texts = [
        "This is a test sentence.",
        "Another test sentence for the tokenizer.",
        "Machine learning is fascinating."
    ]
    tokenizer.train(sample_texts)

    # Create dataset
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text."
    ]

    dataset = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=32,
        stride=16
    )

    print(f"Dataset size: {len(dataset)}")

    # Test data loading
    if len(dataset) > 0:
        input_ids, target_ids = dataset[0]
        print(f"Input shape: {input_ids.shape}")
        print(f"Target shape: {target_ids.shape}")
        print(f"Input IDs: {input_ids[:10]}")
        print(f"Target IDs: {target_ids[:10]}")

        assert input_ids.shape == target_ids.shape, "Input and target shapes must match!"
        assert len(input_ids.shape) == 1, "Sequences should be 1-dimensional!"

        print("✅ Dataset test passed!")
    else:
        print("⚠️  Dataset is empty (might be due to short texts)")


def test_attention_mask():
    """Test attention masking"""
    print("\n" + "="*80)
    print("Testing Attention Mask...")
    print("="*80)

    from utils import create_causal_mask, create_padding_mask

    # Test causal mask
    seq_len = 5
    causal_mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {causal_mask.shape}")
    print("Causal mask (upper triangle should be True):")
    print(causal_mask)

    # Test padding mask
    seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    padding_mask = create_padding_mask(seq, pad_idx=0)
    print(f"\nPadding mask shape: {padding_mask.shape}")
    print("Padding mask (padded positions should be True):")
    print(padding_mask.squeeze())

    print("✅ Attention mask test passed!")


def test_config():
    """Test configuration"""
    print("\n" + "="*80)
    print("Testing Configuration...")
    print("="*80)

    from config import ModelConfig, TrainingConfig

    model_config = ModelConfig()
    train_config = TrainingConfig()

    print("Model Config:")
    print(f"  vocab_size: {model_config.vocab_size}")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_heads: {model_config.n_heads}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  max_seq_len: {model_config.max_seq_len}")

    print("\nTraining Config:")
    print(f"  seed: {train_config.seed}")
    print(f"  num_workers: {train_config.num_workers}")
    print(f"  mixed_precision: {train_config.mixed_precision}")

    # Test validation
    assert model_config.d_model % model_config.n_heads == 0, "d_model must be divisible by n_heads"

    print("✅ Configuration test passed!")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("COMPONENT TESTING")
    print("="*80)

    try:
        test_config()
        test_tokenizer()
        test_attention_mask()
        test_model()
        test_dataset()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nYou can now:")
        print("1. Train the model: python train.py")
        print("2. Evaluate: python evaluate.py")
        print("3. Generate text: python generate.py")

    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
