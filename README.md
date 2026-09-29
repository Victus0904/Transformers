# Transformer from Scratch (PyTorch)

A from-scratch implementation of the encoder–decoder Transformer from
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), written in plain PyTorch without `nn.Transformer`.

## What's implemented

| Component | Class |
|---|---|
| Token embeddings (scaled by √d_model) | `InputEmbeddings` |
| Sinusoidal positional encoding | `PositionalEncoding` |
| Layer normalization | `LayerNormalization` |
| Position-wise feed-forward network | `FeedForwardBlock` |
| Scaled dot-product + multi-head attention | `MultiHeadAttentionBlock` |
| Pre-norm residual connection | `ResidualConnection` |
| Encoder / decoder stacks | `EncoderBlock`, `Encoder`, `DecoderBlock`, `Decoder` |
| Output projection to vocabulary | `ProjectionLayer` |
| Full model + builder with Xavier init | `Transformer`, `build_transformer` |

## Usage

```python
import torch
from transformer import build_transformer

model = build_transformer(
    src_vocab_size=10_000, tgt_vocab_size=10_000,
    src_seq_len=128, tgt_seq_len=128,
    d_model=512, N=6, h=8, dropout=0.1, d_ff=2048,
)

src = torch.randint(0, 10_000, (2, 128))
tgt = torch.randint(0, 10_000, (2, 128))
causal_mask = torch.tril(torch.ones(1, 128, 128)).int()

enc = model.encode(src, src_mask=None)
dec = model.decode(enc, None, tgt, causal_mask)
logits = model.project(dec)   # (batch, tgt_seq_len, tgt_vocab_size)
```

## Requirements

```bash
pip install torch
```
