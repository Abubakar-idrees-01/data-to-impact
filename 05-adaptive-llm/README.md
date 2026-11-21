# Adaptive LLM

A small chat assistant designed for weak, GPU-less hardware. It includes:

- **A tiny GPT built from scratch** with Mixture-of-Experts and early-exit layers, so only part of
  the model runs for each token.
- **A hardware-adaptive inference engine** that profiles the machine, routes each prompt to the
  smallest model that can handle it, and uses quantization and speculative decoding for speed.

**Status:** 🚧 planned
