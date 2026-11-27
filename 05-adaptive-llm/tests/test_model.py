"""The model must be causal, able to learn, and training must resume exactly where it stopped."""

import json

import numpy as np
import pytest
import torch

from tinygpt.model import GPT, GPTConfig
from tinygpt.train import lr_at, main as train_main

TINY = GPTConfig(vocab_size=64, ctx=16, d=32, n_layer=2, n_head=4)


def test_shapes_and_tied_weights():
    m = GPT(TINY)
    logits, loss = m(torch.randint(64, (3, 16)), torch.randint(64, (3, 16)))
    assert logits.shape == (3, 16, 64) and loss.ndim == 0
    assert m.head.weight is m.tok.weight
    # 64×32 tokens + 16×32 positions + per block (4·32² attention + 8·32² ffn + 2 norms) + final norm
    assert m.n_params() == 64 * 32 + 16 * 32 + 2 * (12 * 32 * 32 + 4 * 32) + 2 * 32
    with pytest.raises(ValueError):
        m(torch.randint(64, (1, 17)))


def test_causal_future_tokens_dont_change_the_past():
    torch.manual_seed(0)
    m = GPT(TINY).eval()
    a = torch.randint(64, (1, 16))
    b = a.clone()
    b[0, 10:] = (b[0, 10:] + 1) % 64                # change tokens 10..15
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-6)
    assert not torch.allclose(la[0, 10:], lb[0, 10:])


def test_it_can_memorise_a_tiny_dataset():
    torch.manual_seed(0)
    m = GPT(TINY)
    x = torch.randint(64, (4, 17))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first > 4.0 and loss.item() < 0.3        # from ~ln(64) = 4.16 to near zero


def test_generate_stops_at_the_stop_token():
    torch.manual_seed(0)
    m = GPT(TINY).eval()
    out = m.generate(torch.zeros(1, 1, dtype=torch.long), max_new=40, top_k=1)
    assert out.shape[1] == 41
    stop = out[0, 1].item()
    out2 = m.generate(torch.zeros(1, 1, dtype=torch.long), max_new=40, top_k=1, stop=stop)
    assert out2.shape[1] == 2


def test_lr_schedule():
    assert lr_at(0, 1000, 1e-3, 100) == pytest.approx(1e-5)
    assert lr_at(99, 1000, 1e-3, 100) == pytest.approx(1e-3)
    assert lr_at(1000, 1000, 1e-3, 100) == pytest.approx(1e-4)       # decays to 10%


def make_data(tmp_path):
    d = tmp_path / "tokens"
    d.mkdir()
    rng = np.random.default_rng(0)
    for name, n in [("train", 5000), ("valid", 1000)]:
        rng.integers(0, 64, n, dtype=np.uint16).tofile(d / f"{name}.bin")
    (d / "meta.json").write_text(json.dumps({"vocab_size": 64}))
    return d


def test_resume_gives_exactly_the_same_model(tmp_path):
    data = make_data(tmp_path)
    args = ["--steps", "4", "--batch", "4", "--ctx", "16", "--d", "32", "--n-layer", "2", "--n-head", "4",
            "--eval-every", "2", "--eval-windows", "8", "--threads", "1", "--data", str(data)]
    train_main(["--name", "straight", "--out", str(tmp_path / "a"), *args])
    # same run, stopped after the first checkpoint (2 steps), then resumed
    train_main(["--name", "resumed", "--out", str(tmp_path / "b"), *args, "--max-minutes", "0.000001"])
    log = (tmp_path / "b" / "resumed" / "log.jsonl").read_text().splitlines()
    assert len(log) == 1 and json.loads(log[0])["step"] == 2
    train_main(["--name", "resumed", "--out", str(tmp_path / "b"), *args])
    a = torch.load(tmp_path / "a" / "straight" / "model.pt")["model"]
    b = torch.load(tmp_path / "b" / "resumed" / "model.pt")["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


MOE = GPTConfig(vocab_size=64, ctx=16, d=32, n_layer=2, n_head=4, n_experts=4, top_k=1, expert_mult=4)


def test_moe_counts_and_routing():
    from tinygpt.model import MoE
    torch.manual_seed(0)
    dense, moe = GPT(TINY), GPT(MOE)
    ffn = 8 * 32 * 32                                    # one dense feed-forward: up + down
    assert moe.n_params() == dense.n_params() + 2 * (3 * ffn + 4 * 32)        # 3 extra experts + router, per layer
    assert moe.n_active_params() == dense.n_params() + 2 * 4 * 32             # same compute as dense, plus routers
    logits, loss = moe(torch.randint(64, (3, 16)), torch.randint(64, (3, 16)))
    layer = moe.blocks[0].ffn
    assert isinstance(layer, MoE) and layer.last_counts.sum() == 3 * 16       # every token routed once
    assert moe.aux_loss() >= 2.0 - 1e-5                                       # ≥ 1 per layer, = 1 if perfectly even


def test_moe_top2_mixes_two_experts_and_matches_a_slow_loop():
    torch.manual_seed(0)
    c = GPTConfig(vocab_size=64, ctx=16, d=32, n_layer=1, n_head=4, n_experts=4, top_k=2, expert_mult=2)
    layer = GPT(c).blocks[0].ffn.eval()
    x = torch.randn(2, 5, 32)
    fast = layer(x)
    probs = torch.softmax(layer.router(x.reshape(-1, 32)), -1)
    slow = []
    for i, tok in enumerate(x.reshape(-1, 32)):
        w, e = torch.topk(probs[i], 2)
        w = w / w.sum()
        slow.append(sum(w[j] * layer.experts[e[j]](tok[None])[0] for j in range(2)))
    assert torch.allclose(fast.reshape(-1, 32), torch.stack(slow), atol=1e-5)


def test_load_balancing_loss_is_one_when_even_and_large_when_collapsed():
    from tinygpt.model import MoE
    layer = MoE(GPTConfig(d=8, n_experts=4, top_k=1, expert_mult=1))
    with torch.no_grad():
        layer.router.weight.zero_()
    layer(torch.randn(1, 8, 8))                          # equal probabilities; ties go to expert 0
    assert layer.aux_loss.item() == pytest.approx(1.0)   # f = (1,0,0,0), P = 1/4 each -> 4 · 1/4 = 1
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[0] = 100.0                   # everything strongly to expert 0
    layer(torch.ones(1, 8, 8))
    assert layer.aux_loss.item() == pytest.approx(4.0, rel=1e-3)


def test_moe_trains_and_logs_expert_use(tmp_path):
    data = make_data(tmp_path)
    args = ["--steps", "4", "--batch", "4", "--ctx", "16", "--d", "32", "--n-layer", "2", "--n-head", "4",
            "--n-experts", "4", "--top-k", "2", "--eval-every", "2", "--eval-windows", "8", "--threads", "1",
            "--data", str(data)]
    train_main(["--name", "moe", "--out", str(tmp_path / "a"), *args])
    log = [json.loads(l) for l in (tmp_path / "a" / "moe" / "log.jsonl").read_text().splitlines()]
    assert len(log[-1]["expert_share"]) == 2 and abs(sum(log[-1]["expert_share"][0]) - 1) < 1e-3
    run = json.loads((tmp_path / "a" / "moe" / "run.json").read_text())
    assert run["active_params"] < run["params"]
