"""Capture README screenshots by driving the real app in headless Chrome (Playwright).

    python scripts/screenshots.py            # uses the public sample corpus index

Uses an installed Google Chrome/Chromium (no browser download). Screenshots go to docs/images/.
Public screenshots use the Wikipedia sample corpus only: never private course material.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
PORT = 8599
URL = f"http://localhost:{PORT}"


def chrome_path() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    return None


def wait_for_server(timeout: float = 120) -> None:
    t = time.time()
    while time.time() - t < timeout:
        try:
            urllib.request.urlopen(URL, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Streamlit did not start")


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, STUDYRAG_STORAGE_DIR=str(ROOT / "eval/.cache/sample-structure"),
               PYTHONPATH=str(ROOT / "src"))
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app/streamlit_app.py"),
                               "--server.port", str(PORT), "--server.headless", "true",
                               "--browser.gatherUsageStats", "false", "--theme.base", "light"],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chrome_path(), headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
            page.goto(URL)
            page.get_by_text("Library ·").wait_for(timeout=300_000)       # models + index loaded
            time.sleep(2)

            # 1. Ask: grounded answer with citations, confidence, sources and retrieval trace
            q = "Why do transactional emails matter for marketers?"
            page.get_by_test_id("stChatInputTextArea").fill(q)
            page.keyboard.press("Enter")
            page.locator(".badge").first.wait_for(timeout=600_000)
            time.sleep(1)
            page.get_by_text("Sources (").first.click()
            page.get_by_text("How I found this").first.click()
            time.sleep(1)
            page.screenshot(path=str(OUT / "ask.png"), full_page=True)
            print("ask.png")

            # 2. Refusal
            page.get_by_test_id("stChatInputTextArea").fill("What is a good bounce rate for a landing page?")
            page.keyboard.press("Enter")
            page.get_by_text("couldn't find this").last.wait_for(timeout=120_000)
            time.sleep(1)
            page.screenshot(path=str(OUT / "refusal.png"), full_page=True)
            print("refusal.png")

            # 3. Summarize: instant outline
            page.get_by_role("tab", name="📝 Summarize").click()
            page.get_by_text("Outline (instant").wait_for()
            time.sleep(1)
            page.screenshot(path=str(OUT / "outline.png"), full_page=False)
            print("outline.png")

            # 4. MCQ practice: generate 3, answer the first
            page.get_by_role("tab", name="❓ MCQ Practice").click()
            page.get_by_label("Topic (optional").first.fill("email marketing laws")
            page.get_by_label("How many").first.fill("3")
            page.get_by_role("button", name="Generate MCQs").click()
            page.get_by_text("Q3.").wait_for(timeout=900_000)
            time.sleep(2)
            page.locator('[data-testid="stRadio"]').first.locator("label").nth(1).click()
            time.sleep(2)
            page.screenshot(path=str(OUT / "mcq.png"), full_page=True)
            print("mcq.png")
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
