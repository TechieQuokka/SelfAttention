# Transformer Language Model

PyTorch-based Transformer language model implementation trained on WikiText-2 dataset

## Project Structure

```
SelfAttention/
├── config.py              # Model and training configuration
├── model.py               # Transformer model implementation
├── tokenizer.py           # BPE tokenizer
├── dataset.py             # Dataset preprocessing
├── utils.py               # Utility functions
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── generate.py            # Text generation script
├── download_dataset.py    # Dataset download
├── check_dataset.py       # Dataset verification
└── requirements.txt       # Dependencies
```

## Key Features

### Model Architecture
- **Multi-Head Self-Attention**: 8 attention heads
- **Feed-Forward Networks**: GELU activation
- **Positional Encoding**: Sinusoidal encoding
- **Layer Normalization**: Pre-norm architecture
- **Causal Masking**: Masking for autoregressive generation

### Implementation Details
- Transformer decoder structure (GPT-style)
- BPE (Byte-Pair Encoding) tokenizer
- Mixed precision training (AMP) support
- Gradient clipping and warmup
- Cosine annealing learning rate scheduler
- Early stopping and checkpoint saving

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Dataset

```bash
python download_dataset.py
```

The WikiText-2 dataset will be downloaded to `data/wikitext2/`.

### 2. Verify Dataset

```bash
python check_dataset.py
```

### 3. Train Model

```bash
python train.py
```

**Training Process:**
1. Train tokenizer (BPE, vocab_size=30000)
2. Preprocess data and create DataLoader
3. Train Transformer model
4. Monitor validation performance
5. Save best model checkpoint

**Training Configuration (configurable in config.py):**
- Batch size: 32
- Learning rate: 3e-4
- Epochs: 10
- Model dimension: 512
- Attention heads: 8
- Layers: 6
- Feed-forward dimension: 2048
- Max sequence length: 512

### 4. Evaluate Model

```bash
python evaluate.py
```

Calculates loss and perplexity on the test set.

### 5. Generate Text

```bash
python generate.py
```

**Generation Options:**
- `temp <value>`: Temperature setting (default: 1.0)
- `length <n>`: Number of tokens to generate (default: 100)
- `topk <n>`: Top-k sampling (default: 50)
- `topp <value>`: Nucleus sampling (default: 0.95)

**Example:**
```
Prompt: The history of artificial intelligence
temp 0.8
length 150
```

## Model Architecture

### TransformerLM
```
Token Embedding (vocab_size -> d_model)
    ↓
Positional Encoding
    ↓
N × Transformer Blocks:
    ├── Multi-Head Self-Attention
    ├── Layer Norm
    ├── Residual Connection
    ├── Feed-Forward Network
    ├── Layer Norm
    └── Residual Connection
    ↓
Final Layer Norm
    ↓
Language Model Head (d_model -> vocab_size)
```

### Multi-Head Attention
```
Input (batch, seq_len, d_model)
    ↓
Q, K, V Linear Projections
    ↓
Split into n_heads
    ↓
Scaled Dot-Product Attention
    ↓
Concatenate heads
    ↓
Output Projection
    ↓
Output (batch, seq_len, d_model)
```

## Training Results

Checkpoints are saved in the `checkpoints/` directory:
- `best_model.pt`: Model with the lowest validation loss
- `checkpoint_step_*.pt`: Periodically saved checkpoints

Evaluation results are saved in `checkpoints/evaluation_results.txt`.

## Hyperparameter Tuning

You can adjust the following parameters in `config.py`:

**Model Size:**
- `d_model`: Embedding dimension (small: 256, large: 1024)
- `n_layers`: Number of Transformer layers (small: 4, large: 12)
- `n_heads`: Number of attention heads (must divide d_model evenly)
- `d_ff`: Feed-forward dimension (typically d_model * 4)

**Training:**
- `batch_size`: Batch size (adjust based on GPU memory)
- `learning_rate`: Learning rate (typically 1e-4 ~ 5e-4)
- `num_epochs`: Number of training epochs
- `gradient_clip`: Gradient clipping value

## Performance Optimization

**GPU Memory Optimization:**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Reduce max sequence length

**Training Speed Improvement:**
- Increase `num_workers` (DataLoader)
- Set `pin_memory=True`
- Enable mixed precision training

## Troubleshooting

**CUDA Out of Memory:**
```python
# In config.py
config.batch_size = 16  # Reduce
config.max_seq_len = 256  # Reduce
```

**Training Too Slow:**
```python
# In config.py
train_config.num_workers = 8  # Increase
train_config.mixed_precision = True  # Enable
```

**Perplexity Not Decreasing:**
- Adjust learning rate
- Increase warmup steps
- Increase model size (d_model, n_layers)
- Train for more epochs

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [WikiText-2 Dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
