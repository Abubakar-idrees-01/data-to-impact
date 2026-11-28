// The tiny GPT from project 5, Part A, running in plain JavaScript: tokenizer, transformer, Mixture-of-Experts,
// early exits, sampling. No libraries. The same file runs in the browser and in Node (tests/test_web.py checks it
// against PyTorch on the exported weights).
//
// Weights come from `python -m tinygpt.export_web`: int8 with one float32 scale per row, dequantized at load.

// ---------------------------------------------------------------- tokenizer (byte-level BPE, as tokenizer.py)
const PRETOKEN = /'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/g;
const EOT = "<|endoftext|>";

export class Tokenizer {
  constructor(json) {
    this.merges = json.merges;
    this.ranks = new Map();
    this.vocab = [];
    for (let i = 0; i < 256; i++) this.vocab.push(Uint8Array.of(i));
    this.merges.forEach(([a, b], i) => {
      this.ranks.set(a * 65536 + b, i);
      const x = this.vocab[a], y = this.vocab[b], z = new Uint8Array(x.length + y.length);
      z.set(x); z.set(y, x.length);
      this.vocab.push(z);
    });
    this.specials = new Map((json.specials || [EOT]).map((s, i) => [s, this.vocab.length + i]));
    this.eot = this.specials.get(EOT);
    this.size = this.vocab.length + this.specials.size;
    this.cache = new Map();
    this.enc = new TextEncoder();
    this.dec = new TextDecoder("utf-8", { fatal: false });
  }

  bpe(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    const ids = Array.from(this.enc.encode(word));
    while (ids.length > 1) {
      let best = Infinity, at = -1;                 // the earliest-learned pair; leftmost on ties (as in training)
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      ids.splice(at, 2, 256 + best);
    }
    if (this.cache.size < 100000) this.cache.set(word, ids);
    return ids;
  }

  encode(text) {
    const out = [];
    for (const part of text.split(/(<\|endoftext\|>)/)) {
      if (this.specials.has(part)) out.push(this.specials.get(part));
      else if (part) for (const m of part.matchAll(PRETOKEN)) out.push(...this.bpe(m[0]));
    }
    return out;
  }

  bytesOf(id) {
    if (id < this.vocab.length) return this.vocab[id];
    for (const [s, i] of this.specials) if (i === id) return this.enc.encode(s);
    return new Uint8Array(0);
  }

  decode(ids) {
    const parts = ids.map((i) => this.bytesOf(i));
    const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) { all.set(p, o); o += p.length; }
    return this.dec.decode(all);
  }
}

// ---------------------------------------------------------------- maths
function matvec(W, rows, cols, x, out, xOff = 0) {           // out[r] = Σ_c W[r, c] · x[c]   (W row-major, rows × cols)
  for (let r = 0; r < rows; r++) {
    let s = 0;
    const o = r * cols;
    for (let c = 0; c < cols; c++) s += W[o + c] * x[xOff + c];
    out[r] = s;
  }
  return out;
}

function layerNorm(x, w, b, out, eps = 1e-5) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
  const inv = 1 / Math.sqrt(v / n + eps);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  return out;
}

function erf(x) {                                             // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const gelu = (x) => 0.5 * x * (1 + erf(x * Math.SQRT1_2));   // the exact (erf) GELU, as torch.nn.functional.gelu

function softmaxInPlace(a, n = a.length) {
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (a[i] > m) m = a[i];
  let s = 0;
  for (let i = 0; i < n; i++) { a[i] = Math.exp(a[i] - m); s += a[i]; }
  for (let i = 0; i < n; i++) a[i] /= s;
  return a;
}

function topProb(logits) {                                    // max softmax probability, without a copy
  let m = -Infinity, at = 0;
  for (let i = 0; i < logits.length; i++) if (logits[i] > m) { m = logits[i]; at = i; }
  let s = 0;
  for (let i = 0; i < logits.length; i++) s += Math.exp(logits[i] - m);
  return { p: 1 / s, token: at };
}

// ---------------------------------------------------------------- the model
export class TinyGPT {
  /** manifest: the exported <name>.json; buffer: the ArrayBuffer of <name>.bin */
  constructor(manifest, buffer) {
    this.manifest = manifest;
    const c = (this.config = manifest.config);
    this.d = c.d; this.L = c.n_layer; this.H = c.n_head; this.dh = c.d / c.n_head; this.ctx = c.ctx; this.V = c.vocab_size;
    this.W = {};
    for (const [name, t] of Object.entries(manifest.tensors)) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      if (t.dtype === "int8") {
        const q = new Int8Array(buffer, t.offset, n), scale = new Float32Array(buffer, t.scale_offset, t.shape[0]);
        const w = new Float32Array(n), cols = t.shape[1];
        for (let r = 0; r < t.shape[0]; r++) for (let j = 0; j < cols; j++) w[r * cols + j] = q[r * cols + j] * scale[r];
        this.W[name] = w;
      } else {
        this.W[name] = new Float32Array(buffer.slice(t.offset, t.offset + 4 * n));
      }
    }
    this.moe = c.n_experts > 0;
    this.hasExits = !!this.W["exit_norms.0.weight"];
    this.hidden = this.moe ? c.expert_mult * c.d : c.ff_mult * c.d;
    // costs in multiply-adds, for the "compute used" counter
    this.macsLayerFixed = 4 * c.d * c.d + 2 * c.d * this.hidden * (this.moe ? c.top_k : 1) + (this.moe ? c.n_experts * c.d : 0);
    this.macsHead = c.vocab_size * c.d;
    // buffers
    const d = c.d;
    this.x = new Float32Array(d); this.h = new Float32Array(d); this.qkv = new Float32Array(3 * d);
    this.att = new Float32Array(d); this.proj = new Float32Array(d); this.up = new Float32Array(this.hidden);
    this.ffo = new Float32Array(d); this.acc = new Float32Array(d); this.scores = new Float32Array(c.ctx);
    this.logits = new Float32Array(c.vocab_size); this.router = new Float32Array(Math.max(1, c.n_experts));
    this.K = Array.from({ length: c.n_layer }, () => new Float32Array(c.ctx * d));
    this.Vc = Array.from({ length: c.n_layer }, () => new Float32Array(c.ctx * d));
    this.reset();
  }

  reset() { this.t = 0; }

  head(normW, normB, x) {                                     // next-token scores: tied embedding · norm(x)
    layerNorm(x, normW, normB, this.h);
    return matvec(this.W["tok.weight"], this.V, this.d, this.h, this.logits);
  }

  attention(i, x) {                                           // x += out(attention(norm1(x))), caching k, v at t
    const d = this.d, W = this.W, p = `blocks.${i}.`, t = this.t, dh = this.dh;
    layerNorm(x, W[p + "norm1.weight"], W[p + "norm1.bias"], this.h);
    matvec(W[p + "attn.qkv.weight"], 3 * d, d, this.h, this.qkv);
    const K = this.K[i], Vc = this.Vc[i];
    K.set(this.qkv.subarray(d, 2 * d), t * d);
    Vc.set(this.qkv.subarray(2 * d, 3 * d), t * d);
    const scale = 1 / Math.sqrt(dh);
    for (let hd = 0; hd < this.H; hd++) {
      const o = hd * dh;
      for (let j = 0; j <= t; j++) {
        let s = 0;
        for (let k = 0; k < dh; k++) s += this.qkv[o + k] * K[j * d + o + k];
        this.scores[j] = s * scale;
      }
      softmaxInPlace(this.scores, t + 1);
      for (let k = 0; k < dh; k++) {
        let s = 0;
        for (let j = 0; j <= t; j++) s += this.scores[j] * Vc[j * d + o + k];
        this.att[o + k] = s;
      }
    }
    matvec(W[p + "attn.out.weight"], d, d, this.att, this.proj);
    for (let k = 0; k < d; k++) x[k] += this.proj[k];
  }

  mlp(prefix, h, out) {                                       // down(gelu(up(h)))
    const d = this.d, n = this.hidden;
    matvec(this.W[prefix + "up.weight"], n, d, h, this.up);
    for (let k = 0; k < n; k++) this.up[k] = gelu(this.up[k]);
    return matvec(this.W[prefix + "down.weight"], d, n, this.up, out);
  }

  feedForward(i, x) {                                         // x += ffn(norm2(x)); returns the chosen experts (MoE)
    const W = this.W, p = `blocks.${i}.`, d = this.d;
    layerNorm(x, W[p + "norm2.weight"], W[p + "norm2.bias"], this.h);
    if (!this.moe) {
      this.mlp(p + "ffn.", this.h, this.ffo);
      for (let k = 0; k < d; k++) x[k] += this.ffo[k];
      return null;
    }
    const E = this.config.n_experts, topK = this.config.top_k;
    matvec(W[p + "ffn.router.weight"], E, d, this.h, this.router);
    softmaxInPlace(this.router, E);
    const order = Array.from({ length: E }, (_, e) => e).sort((a, b) => this.router[b] - this.router[a]).slice(0, topK);
    let norm = 0;
    for (const e of order) norm += this.router[e];
    this.acc.fill(0);
    const chosen = [];
    for (const e of order) {
      const w = topK > 1 ? this.router[e] / norm : this.router[e];   // top-1: raw p (Switch); top-k: renormalised
      this.mlp(`${p}ffn.experts.${e}.`, this.h, this.ffo);
      for (let k = 0; k < d; k++) this.acc[k] += w * this.ffo[k];
      chosen.push({ expert: e, weight: w });
    }
    for (let k = 0; k < d; k++) x[k] += this.acc[k];
    return chosen;
  }

  fillCache(i, x) {                                           // early exit: layer i's k, v from the exited state (CALM)
    const d = this.d, W = this.W, p = `blocks.${i}.`;
    layerNorm(x, W[p + "norm1.weight"], W[p + "norm1.bias"], this.h);
    matvec(W[p + "attn.qkv.weight"], 3 * d, d, this.h, this.qkv);
    this.K[i].set(this.qkv.subarray(d, 2 * d), this.t * d);
    this.Vc[i].set(this.qkv.subarray(2 * d, 3 * d), this.t * d);
  }

  /**
   * Feed one token at position t; return the next-token scores.
   * opts.exits: also report every layer's exit (top token, its probability)
   * opts.exitThreshold: stop at the first layer whose exit is this confident (exit models only); the skipped layers
   *   get keys/values computed from the exited state so later tokens can still attend to this one (CALM)
   */
  step(token, opts = {}) {
    if (this.t >= this.ctx) throw new Error(`context full (${this.ctx} tokens)`);
    const d = this.d, W = this.W, x = this.x;
    for (let k = 0; k < d; k++) x[k] = W["tok.weight"][token * d + k] + W["pos.weight"][this.t * d + k];
    const info = { experts: [], exits: [], layers: this.L, macs: 0 };
    const attnMacs = 2 * (this.t + 1) * d;
    const early = this.hasExits && opts.exitThreshold != null && opts.exitThreshold <= 1;
    for (let i = 0; i < this.L; i++) {
      this.attention(i, x);
      const chosen = this.feedForward(i, x);
      info.macs += this.macsLayerFixed + attnMacs;
      if (chosen) info.experts.push(chosen);
      const last = i === this.L - 1;
      if (!last && (opts.exits || early) && (this.hasExits || opts.exits)) {
        const nw = this.hasExits ? W[`exit_norms.${i}.weight`] : W["norm.weight"];   // untrained: the logit lens
        const nb = this.hasExits ? W[`exit_norms.${i}.bias`] : W["norm.bias"];
        this.head(nw, nb, x);
        info.macs += this.macsHead;
        const tp = topProb(this.logits);
        info.exits.push(tp);
        if (early && tp.p >= opts.exitThreshold) {
          for (let j = i + 1; j < this.L; j++) { this.fillCache(j, x); info.macs += 3 * d * d; }
          info.layers = i + 1;
          this.t++;
          return { logits: this.logits, ...info };
        }
      }
    }
    this.head(W["norm.weight"], W["norm.bias"], x);
    info.macs += this.macsHead;
    if (opts.exits) info.exits.push(topProb(this.logits));
    this.t++;
    return { logits: this.logits, ...info };
  }

  /** Full-model cost of one token at position t, in multiply-adds (for the "compute used" counter). */
  fullMacs(t) { return this.L * (this.macsLayerFixed + 2 * (t + 1) * this.d) + this.macsHead; }
}

// ---------------------------------------------------------------- sampling
export function mulberry32(seed) {                            // small seeded RNG, so a seed gives the same story
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Temperature + top-k sampling (temperature 0 = greedy). Returns { token, p } with p under the sampled distribution. */
export function sample(logits, { temperature = 0.8, topK = 40, rng = Math.random } = {}) {
  const n = logits.length;
  if (temperature <= 0) { const t = topProb(logits); return { token: t.token, p: t.p }; }
  const idx = Array.from({ length: n }, (_, i) => i);
  const k = Math.min(topK || n, n);
  idx.sort((a, b) => logits[b] - logits[a]);
  const keep = idx.slice(0, k), probs = new Float64Array(k);
  for (let i = 0; i < k; i++) probs[i] = logits[keep[i]] / temperature;
  softmaxInPlace(probs, k);
  let r = rng(), i = 0;
  for (; i < k - 1; i++) { r -= probs[i]; if (r <= 0) break; }
  return { token: keep[i], p: probs[i] };
}
