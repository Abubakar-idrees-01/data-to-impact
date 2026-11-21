"""Fetch the public sample corpus: Wikipedia articles on digital marketing (CC BY-SA 4.0).

These are used for the public demo and the public evaluation set, because real course slides
are copyrighted and must not be published. Output: data/sample_docs/*.md
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_docs"

ARTICLES = [
    "Digital marketing",
    "Search engine optimization",
    "Search engine marketing",
    "Social media marketing",
    "Email marketing",
    "Content marketing",
    "Affiliate marketing",
    "Pay-per-click",
    "Influencer marketing",
    "Marketing mix",
]

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "StudyRAG-sample-fetcher/0.1 (educational open-source project)"}
SKIP_SECTIONS = {"see also", "references", "further reading", "external links", "notes", "bibliography", "sources"}


def to_markdown(title: str, text: str) -> str:
    lines, skipping = [], False
    for line in text.splitlines():
        m = re.match(r"^(=+)\s*(.*?)\s*=+$", line.strip())
        if m:
            level, heading = len(m.group(1)), m.group(2)
            skipping = heading.lower() in SKIP_SECTIONS or (skipping and level > 2)
            if not skipping:
                lines.append(f"\n{'#' * level} {heading}\n")
            continue
        if not skipping:
            lines.append(line)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    return (f"# {title}\n\n{body}\n\n---\n*Source: Wikipedia, \"{title}\" ({url}), "
            "licensed under CC BY-SA 4.0. Text extracted without modification apart from formatting.*\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, timeout=60) as client:
        for title in ARTICLES:
            r = client.get(API, params={"action": "query", "prop": "extracts", "explaintext": 1,
                                        "exsectionformat": "wiki", "titles": title, "format": "json",
                                        "redirects": 1})
            r.raise_for_status()
            page = next(iter(r.json()["query"]["pages"].values()))
            md = to_markdown(page["title"], page.get("extract", ""))
            path = OUT / (page["title"].replace(" ", "_").replace("/", "-") + ".md")
            path.write_text(md, encoding="utf-8")
            print(f"{path.name}: {len(md.split()):,} words")


if __name__ == "__main__":
    main()
