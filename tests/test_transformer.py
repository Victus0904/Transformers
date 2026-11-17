import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer import build_transformer, causal_mask  # noqa: E402

SRC_V, TGT_V, SEQ = 50, 60, 12


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_transformer(SRC_V, TGT_V, SEQ, SEQ, d_model=32, N=2, h=4, d_ff=64, dropout=0.0)


def test_forward_shapes(model):
    src = torch.randint(0, SRC_V, (3, SEQ))
    tgt = torch.randint(0, TGT_V, (3, SEQ))
    enc = model.encode(src, None)
    assert enc.shape == (3, SEQ, 32)
    dec = model.decode(enc, None, tgt, causal_mask(SEQ))
    assert model.project(dec).shape == (3, SEQ, TGT_V)


def test_causal_mask_blocks_future_tokens(model):
    """Changing a later target token must not change outputs at earlier positions."""
    model.eval()
    src = torch.randint(0, SRC_V, (1, SEQ))
    tgt = torch.randint(0, TGT_V, (1, SEQ))
    tgt2 = tgt.clone()
    tgt2[0, -1] = (tgt2[0, -1] + 1) % TGT_V
    enc = model.encode(src, None)
    out1 = model.decode(enc, None, tgt, causal_mask(SEQ))
    out2 = model.decode(enc, None, tgt2, causal_mask(SEQ))
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
    assert not torch.allclose(out1[:, -1], out2[:, -1])


def test_padding_mask_ignores_padded_source(model):
    model.eval()
    src = torch.randint(1, SRC_V, (1, SEQ))
    src2 = src.clone()
    src2[0, -3:] = 0  # pretend these are padding tokens
    src_mask = torch.ones(1, 1, 1, SEQ, dtype=torch.int)
    src_mask[..., -3:] = 0
    tgt = torch.randint(0, TGT_V, (1, SEQ))
    outs = [
        model.decode(model.encode(s, src_mask), src_mask, tgt, causal_mask(SEQ))
        for s in (src, src2)
    ]
    # Encoder outputs at padded positions differ, but the decoder never attends to them.
    assert torch.allclose(outs[0], outs[1], atol=1e-5)


def test_training_step_reduces_loss(model):
    torch.manual_seed(0)
    src = torch.randint(0, SRC_V, (8, SEQ))
    tgt = src.clone() % TGT_V  # copy task
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    def step():
        logits = model.project(model.decode(model.encode(src, None), None, tgt, causal_mask(SEQ)))
        return loss_fn(logits.reshape(-1, TGT_V), tgt.reshape(-1))

    first = step().item()
    for _ in range(30):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
    assert loss.item() < first
