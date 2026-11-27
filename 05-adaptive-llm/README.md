# ⚡ Adaptive LLM: a tiny GPT from scratch, and an engine that fits models to weak hardware

A chat assistant designed for weak, GPU-less machines, built in two parts:

- **A tiny GPT built from scratch** (tokenizer, transformer, training loop) with **Mixture-of-Experts** and
  **early-exit** layers, so only part of the model runs for each token, trained on a 2015 laptop CPU.
- **A hardware-adaptive inference engine** that profiles the machine, routes each prompt to the smallest model that
  can handle it, and uses quantization and speculative decoding for speed.

**Status:** 🚧 Part A in progress. A1 (data, tokenizer) and A2 (dense baseline) done; MoE and early-exit runs training.

## Why
Training a ChatGPT-quality model is impossible on a CPU. Part A builds a tiny model to *show* the ideas behind
efficient models and measure them honestly; Part B applies them to real open models people can actually use.

## Progress
| phase | what | status |
|---|---|---|
| A1 | TinyStories data + byte-level BPE tokenizer written from scratch | ✅ 4,096 tokens, 57.6M tokens of stories, 3.95 chars/token |
| A2 | dense GPT baseline, training loop, CPU throughput | ✅ 5.84M params, 12.3M tokens in 96 min, val loss 2.235 (ppl 9.3); 2 threads beat 4 |
| A3 | Mixture-of-Experts layers | code + tests done; training |
| A4 | early exit | code + tests done; queued |
| B1 | hardware profiler (cores, SIMD flags, RAM, GPU, measured bandwidth), GGUF reader, model/quant planner | code + 11 tests done; measurements after Part A (needs an idle CPU) |
| B2 | llama.cpp runtime (streaming, TTFT/decode timing, KV-cache reuse), model ladder with LRU unloading, benchmark + calibration | code + 5 tests done; measurements after Part A |
| B3–B5 | complexity router + cascade, speculative decoding, benchmarks via evalkit | |

## Quickstart
```bash
cd 05-adaptive-llm
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt                  # CPU-only torch, numpy, pytest
bash scripts/get_data.sh                                   # TinyStories: 22 MB valid + first 200 MB of train
PYTHONPATH=src .venv/bin/python -m tinygpt.data            # train the tokenizer, encode the stories (~1 min)
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m adaptive.plan           # what can this machine run? (Part B)
```
Full guide: [docs/SETUP.md](docs/SETUP.md)

## Documentation
| | |
|---|---|
| [Setup](docs/SETUP.md) | Install on Linux / Windows / macOS, data, troubleshooting |
| Concepts | 1. [Tokenization (BPE)](docs/concepts/01-tokenization-bpe.md) · 2. [The transformer and training](docs/concepts/02-the-transformer-and-training.md) · 3. [Mixture-of-Experts](docs/concepts/03-mixture-of-experts.md) · 5. [Hardware-aware model choice](docs/concepts/05-hardware-aware-model-choice.md) |

## Data
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023), V2 stories written by
GPT-4, licence CDLA-Sharing-1.0. Downloaded by script, not stored in this repo.

## License
MIT.
