// Runs site/tinygpt.js on reference cases written by tests/test_web.py and prints how far it is from PyTorch.
//   node tests/web_check.mjs <path to tinygpt.js> <cases.json>
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [modPath, casesPath] = process.argv.slice(2);
const { TinyGPT, Tokenizer } = await import(pathToFileURL(modPath).href);
const cases = JSON.parse(readFileSync(casesPath, "utf8"));
const out = { models: [], tokenizer: null };

for (const c of cases.models) {
  const manifest = JSON.parse(readFileSync(c.manifest, "utf8"));
  const bin = readFileSync(c.bin);
  const m = new TinyGPT(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
  let maxLogit = 0, maxExit = 0, argmaxSame = 0;
  c.tokens.forEach((tok, t) => {
    const r = m.step(tok, { exits: c.exit_top_probs != null });
    const ref = c.logits[t];
    let best = 0;
    for (let v = 0; v < ref.length; v++) {
      maxLogit = Math.max(maxLogit, Math.abs(r.logits[v] - ref[v]));
      if (r.logits[v] > r.logits[best]) best = v;
    }
    if (best === c.argmax[t]) argmaxSame++;
    if (c.exit_top_probs) r.exits.forEach((e, i) => { maxExit = Math.max(maxExit, Math.abs(e.p - c.exit_top_probs[t][i])); });
  });
  out.models.push({ name: c.name, max_logit_diff: maxLogit, max_exit_prob_diff: maxExit, argmax_same: argmaxSame / c.tokens.length });
}

if (cases.tokenizer) {
  const tok = new Tokenizer(JSON.parse(readFileSync(cases.tokenizer.path, "utf8")));
  out.tokenizer = cases.tokenizer.texts.map((text, i) => {
    const ids = tok.encode(text);
    return { same_ids: JSON.stringify(ids) === JSON.stringify(cases.tokenizer.ids[i]), round_trip: tok.decode(ids) === text };
  });
}
console.log(JSON.stringify(out));
