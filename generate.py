"""
Text generation script
"""
import torch
from pathlib import Path
from model import TransformerLM
from tokenizer import SimpleTokenizer
from config import ModelConfig

@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda"
):
    """
    Generate text from a prompt

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Starting text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling
        top_p: Nucleus sampling threshold
        device: Device to run on

    Returns:
        Generated text
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...")
    print("-" * 80)

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    print(f"Generated text:\n{generated_text}")
    print("-" * 80)

    return generated_text


def load_model_for_generation(checkpoint_path: str, config: ModelConfig, device: str = "cuda"):
    """Load trained model for generation"""
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
    print(f"Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}")

    return model, tokenizer


def interactive_generation(model: TransformerLM, tokenizer: SimpleTokenizer, device: str = "cuda"):
    """Interactive text generation"""
    print("\n" + "=" * 80)
    print("Interactive Text Generation")
    print("=" * 80)
    print("Enter a prompt to generate text (or 'quit' to exit)")
    print("Commands:")
    print("  temp <value>  - Set temperature (default: 1.0)")
    print("  length <n>    - Set max tokens to generate (default: 100)")
    print("  topk <n>      - Set top-k value (default: 50)")
    print("  topp <value>  - Set top-p value (default: 0.95)")
    print("=" * 80)

    temperature = 1.0
    max_tokens = 100
    top_k = 50
    top_p = 0.95

    while True:
        try:
            prompt = input("\nPrompt: ").strip()

            if not prompt:
                continue

            if prompt.lower() == 'quit':
                break

            # Handle commands
            if prompt.startswith('temp '):
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
                continue
            elif prompt.startswith('length '):
                max_tokens = int(prompt.split()[1])
                print(f"Max tokens set to {max_tokens}")
                continue
            elif prompt.startswith('topk '):
                top_k = int(prompt.split()[1])
                print(f"Top-k set to {top_k}")
                continue
            elif prompt.startswith('topp '):
                top_p = float(prompt.split()[1])
                print(f"Top-p set to {top_p}")
                continue

            # Generate text
            generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device
            )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


def main():
    """Main generation function"""
    config = ModelConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return

    model, tokenizer = load_model_for_generation(str(checkpoint_path), config, device)

    # Example generations
    prompts = [
        "The history of artificial intelligence",
        "In the world of technology,",
        "Scientists have discovered",
    ]

    print("\n" + "=" * 80)
    print("Example Generations")
    print("=" * 80)

    for prompt in prompts:
        generate_text(model, tokenizer, prompt, max_new_tokens=100, device=device)

    # Interactive mode
    interactive_generation(model, tokenizer, device)


if __name__ == "__main__":
    main()
