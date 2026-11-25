"""Web tools with a fake internet (no network, no key spent)."""

import asyncio
import json

import httpx
import pytest

from scout.tools import ToolError
from scout.web import SourceBook, Web, check_public_url, focus, html_to_text

ARTICLE = """<html><head><title>Backlinks explained</title><script>var x = "ignore me";</script></head><body>
<nav>Home | About</nav><h1>Backlinks</h1><p>Search engines treat quality backlinks from other websites as votes of trust.</p>
<p>Unrelated paragraph about cooking pasta with tomatoes and basil for a quick dinner tonight, with garlic, olive oil and a little parmesan on top.</p>
<footer>Copyright</footer></body></html>"""


def fake_web(routes: dict, **kw) -> tuple[Web, list]:
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.method, str(req.url), req.content))
        status, body, headers = routes[str(req.url)]
        if isinstance(body, dict):
            return httpx.Response(status, json=body, headers=headers)
        return httpx.Response(status, text=body, headers=headers)
    return Web("tvly-test", transport=httpx.MockTransport(handler), url_check=lambda u: u, **kw), seen


SEARCH = {"results": [{"title": "Backlinks explained", "url": "https://seo.example/backlinks#top", "content": "votes of trust " * 3},
                      {"title": "Other", "url": "https://other.example/", "content": "x"}]}


def test_search_numbers_sources_and_read_keeps_the_number():
    web, seen = fake_web({"https://api.tavily.com/search": (200, SEARCH, {}),
                          "https://seo.example/backlinks": (200, ARTICLE, {"content-type": "text/html"})})
    res = asyncio.run(web.search("what are backlinks"))
    assert [r["n"] for r in res["results"]] == [1, 2] and res["results"][0]["url"] == "https://seo.example/backlinks"
    assert json.loads(seen[0][2])["query"] == "what are backlinks"
    page = asyncio.run(web.read_page("1", "backlinks"))
    assert page["n"] == 1 and "votes of trust" in page["text"] and "ignore me" not in page["text"]
    assert "Home | About" not in page["text"] and web.book.get(1).read
    assert [s.n for s in web.book.cited("Backlinks are votes [1].")] == [1]


def test_read_falls_back_to_tavily_extract_when_blocked():
    web, _ = fake_web({"https://blocked.example/a": (403, "Forbidden", {}),
                       "https://api.tavily.com/extract": (200, {"results": [{"raw_content": "The real text. " * 30}]}, {})})
    page = asyncio.run(web.read_page("https://blocked.example/a"))
    assert page["n"] == 1 and "The real text." in page["text"]


def test_redirects_are_checked_hop_by_hop():
    def check(url):
        if "169.254" in url:
            raise ToolError("local and private network addresses can't be read")
        return url
    web, seen = fake_web({"https://short.example/x": (302, "", {"location": "http://169.254.169.254/latest/meta-data"})})
    web.url_check = check
    with pytest.raises(ToolError, match="private"):
        asyncio.run(web.read_page("https://short.example/x"))
    assert all("169.254" not in url for _, url, _ in seen)      # the private address was never requested


def test_public_url_check():
    for bad in ("file:///etc/passwd", "http://localhost:8000", "http://127.0.0.1/", "http://10.0.0.5/x",
                "http://192.168.1.1/admin", "ftp://example.com", "http://printer.local/"):
        with pytest.raises(ToolError):
            check_public_url(bad)


def test_focus_keeps_relevant_passages_in_order():
    paras = [f"Paragraph {i} talks about gardening and soil." for i in range(40)]
    paras[7] = "The Model Context Protocol was announced by Anthropic in November 2024."
    paras[30] = "MCP servers expose tools, resources and prompts over JSON-RPC."
    text, truncated = focus("\n\n".join(paras), "When was the Model Context Protocol announced? MCP servers", 300)
    assert truncated and text.index("announced") < text.index("JSON-RPC") and "gardening" not in text


def test_html_to_text_title_and_blocks():
    title, text = html_to_text(ARTICLE)
    assert title == "Backlinks explained" and "## Backlinks" in text and "Copyright" not in text


def test_search_errors_are_explained():
    web, _ = fake_web({"https://api.tavily.com/search": (401, {"detail": "bad"}, {})})
    with pytest.raises(ToolError, match="invalid or exhausted"):
        asyncio.run(web.search("x y"))
    with pytest.raises(ToolError, match="not set"):
        asyncio.run(Web("").search("x y"))
    with pytest.raises(ToolError, match="no source"):
        asyncio.run(Web("k").read_page("9"))


def test_source_book_dedupes_urls():
    b = SourceBook()
    assert b.add("https://a.example/x#frag").n == b.add("https://a.example/x/").n == 1 and b.add("https://b.example").n == 2
