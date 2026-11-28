"""Run the browser demo headlessly and capture README screenshots (Playwright + the installed Chrome).

    python -m tinygpt.export_web             # once: writes site/models/ and site/tokenizer.json
    python scripts/screenshots.py            # serves site/, writes docs/images/web-*.png

Every shot waits until the page has written its story (`body[data-ready]`) and fails on any JavaScript error, so
this doubles as an end-to-end test of the real page with the real weights.
"""

from __future__ import annotations

import functools
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE, OUT = ROOT / "site", ROOT / "docs" / "images"

SHOTS = [  # (file, query string, what it shows)
    ("web-moe.png", "model=moe-8x-top2&seed=1&colour=expert&run", "MoE: tokens coloured by the expert chosen in layer 1"),
    ("web-early-exit.png", "model=exit-d256-L6&seed=1&early=0.6&colour=exit&run",
     "early exit at τ = 0.6: layers per token and the real compute"),
    ("web-dense.png", "model=dense-d256-L6&seed=1&colour=prob&run", "dense: tokens shaded by the chosen word's probability"),
]


def chrome() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    return None


def main() -> None:
    from playwright.sync_api import sync_playwright

    if not (SITE / "models" / "index.json").exists():
        raise SystemExit("no exported models: run `python -m tinygpt.export_web` first")
    OUT.mkdir(parents=True, exist_ok=True)
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(SITE))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/index.html"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chrome(), headless=True)
            for name, query, what in SHOTS:
                page = browser.new_page(viewport={"width": 1280, "height": 1000}, color_scheme="light")
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                page.goto(f"{base}?{query}")
                page.wait_for_selector("body[data-ready='1']", timeout=180_000)
                if errors:
                    raise SystemExit(f"{name}: JavaScript errors: {errors}")
                stats = {k: page.inner_text(f"#s-{k}") for k in ("speed", "tokens", "layers", "compute")}
                page.screenshot(path=str(OUT / name), full_page=False)
                print(f"{name:20s} {what}: {stats}")
                page.close()
            browser.close()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
