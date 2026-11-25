"""Keeping the conversation inside the model's request limit.

Every model call resends the whole conversation: system prompt, task, every tool call and every result. After a
few web pages that grows past what one request may carry (on Groq's free tier, 8,000 tokens per minute, so one
request can't be bigger than that). The first live digest failed exactly like this: "Requested 9042, Limit 8000".

Strategy (the simplest that works):
1. Estimate the request size before sending it. Characters per token is calibrated after every call from the
   real `prompt_tokens` the API reports, so the estimate tracks the actual tokenizer.
2. If it won't fit, shrink the **oldest** tool results first to a short stub (their first few hundred characters:
   for a web page that's its number, title and URL), keeping the most recent results whole. The model's own
   messages (its plan, what it concluded) are never trimmed.
3. If the provider still says "too large", lower the limit and try again.
"""

from __future__ import annotations

import json

STUB = "\n...[trimmed to save context; repeat the call if you need the full result again]"


class ContextWindow:
    def __init__(self, limit_tokens: int | None, *, keep_recent: int = 2, head_chars: int = 300,
                 chars_per_token: float = 3.5):
        self.limit = limit_tokens
        self.keep_recent = keep_recent
        self.head_chars = head_chars
        self.cpt = chars_per_token
        self.trimmed = 0

    @staticmethod
    def size(messages: list[dict], tools: list[dict] | None) -> int:
        return len(json.dumps(messages, ensure_ascii=False)) + (len(json.dumps(tools, ensure_ascii=False)) if tools else 0)

    def estimate(self, messages: list[dict], tools: list[dict] | None) -> int:
        return int(self.size(messages, tools) / self.cpt)

    def calibrate(self, messages: list[dict], tools: list[dict] | None, prompt_tokens: int) -> None:
        if prompt_tokens > 200:
            self.cpt = max(1.5, min(6.0, self.size(messages, tools) / prompt_tokens))

    def fit(self, messages: list[dict], tools: list[dict] | None, reserve: int) -> int:
        """Trim old tool results in place until estimate + reserve <= limit. Returns how many were trimmed."""
        if not self.limit:
            return 0
        tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        candidates = tool_idx[:-self.keep_recent] if self.keep_recent else tool_idx
        n = 0
        for i in candidates:
            if self.estimate(messages, tools) + reserve <= self.limit:
                break
            content = messages[i]["content"]
            if content.endswith(STUB) or len(content) <= self.head_chars + len(STUB):
                continue
            messages[i] = {**messages[i], "content": content[: self.head_chars] + STUB}
            n += 1
        if self.estimate(messages, tools) + reserve > self.limit and self.keep_recent:
            # still too big: the recent results have to shrink too (all but the newest)
            for i in tool_idx[-self.keep_recent:-1]:
                content = messages[i]["content"]
                if not content.endswith(STUB) and len(content) > self.head_chars * 3:
                    messages[i] = {**messages[i], "content": content[: self.head_chars * 3] + STUB}
                    n += 1
        self.trimmed += n
        return n
