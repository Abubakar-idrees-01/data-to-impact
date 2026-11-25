"""The loop, the registry and the approval gate, with a scripted model."""

import asyncio
import json

from conftest import calls, text
from scout.agent import Agent
from scout.approval import Action, DenyAll, Scripted
from scout.tools import Registry, Tool, ToolError

ADD = Tool("add", "Add two integers.", {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                        "required": ["a", "b"]}, lambda a, b: {"sum": a + b})


def run(agent, task="task"):
    return asyncio.run(agent.run(task))


def test_loop_calls_tool_then_answers(fake_api, tmp_path):
    api = fake_api([calls(("add", {"a": 2, "b": 3})), text("The sum is 5.")])
    r = run(Agent(api.llm(), Registry().add(ADD), traces_dir=tmp_path))
    assert r.answer == "The sum is 5." and r.stopped == "answer" and r.steps == 2 and r.tool_calls == 1
    tool_msg = api.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"]) == {"sum": 5}
    assert api.requests[0]["reasoning_effort"] == "low" and api.requests[0]["tools"][0]["function"]["name"] == "add"
    assert r.prompt_tokens == 200 and r.completion_tokens == 40
    lines = [json.loads(l) for l in open(r.trace_path)]
    assert [e["type"] for e in lines] == ["task", "llm", "tool", "llm", "done"]


def test_bad_arguments_go_back_to_the_model(fake_api):
    api = fake_api([calls(("add", {"a": "two", "b": 3})), calls(("nope", {})), calls(("add", {"a": 2})), text("ok")])
    r = run(Agent(api.llm(), Registry().add(ADD)))
    results = [e["result"] for e in r.events if e["type"] == "tool"]
    assert "a must be a integer" in results[0] and "no tool named 'nope'" in results[1] and "missing required" in results[2]
    assert r.tool_errors == 3 and r.answer == "ok"


def test_step_limit_forces_a_final_answer_without_tools(fake_api):
    api = fake_api([calls(("add", {"a": i, "b": 1})) for i in range(3)] + [text("best effort")])
    r = run(Agent(api.llm(), Registry().add(ADD), max_steps=3))
    assert r.stopped == "max_steps" and r.answer == "best effort" and r.steps == 4
    assert "tools" not in api.requests[-1] and "step budget" in api.requests[-1]["messages"][-1]["content"]


def test_token_budget_stops_early(fake_api):
    api = fake_api([calls(("add", {"a": 1, "b": 1}), prompt=900), text("summary")])
    r = run(Agent(api.llm(), Registry().add(ADD), token_budget=500))
    assert r.stopped == "budget" and r.answer == "summary" and r.steps == 2


def test_repeated_identical_call_is_not_run_twice(fake_api):
    hits = []
    tool = Tool("ping", "p", {"type": "object", "properties": {}}, lambda: hits.append(1) or "pong")
    api = fake_api([calls(("ping", {})), calls(("ping", {})), text("done")])
    r = run(Agent(api.llm(), Registry().add(tool)))
    assert len(hits) == 1 and r.tool_calls == 1 and "already made this exact call" in api.requests[2]["messages"][-1]["content"]


def _outward(ran):
    return Tool("push", "Push code.", {"type": "object", "properties": {"branch": {"type": "string"}}, "required": ["branch"]},
                lambda branch: ran.append(branch) or "pushed", outward=True,
                preview=lambda a: Action("push", f"push to {a['branch']}", "diff --git ..."))


def test_outward_tool_never_runs_without_approval(fake_api):
    ran, approver = [], DenyAll()
    api = fake_api([calls(("push", {"branch": "main"})), text("I could not push.")])
    r = run(Agent(api.llm(), Registry(approver).add(_outward(ran))))
    assert ran == [] and r.declined == 1 and approver.seen[0].summary == "push to main"
    assert "declined" in api.requests[1]["messages"][-1]["content"]


def test_outward_tool_runs_after_approval(fake_api):
    ran = []
    api = fake_api([calls(("push", {"branch": "main"})), text("Pushed.")])
    r = run(Agent(api.llm(), Registry(Scripted([True])).add(_outward(ran))))
    assert ran == ["main"] and r.declined == 0


def test_tool_errors_and_crashes_are_reported_not_raised(fake_api):
    def boom():
        raise RuntimeError("disk on fire")
    def expected():
        raise ToolError("file not found")
    reg = Registry().add(Tool("boom", "b", {"type": "object", "properties": {}}, boom),
                         Tool("exp", "e", {"type": "object", "properties": {}}, expected))
    api = fake_api([calls(("boom", {}), ("exp", {})), text("sorry")])
    r = run(Agent(api.llm(), reg))
    res = [e["result"] for e in r.events if e["type"] == "tool"]
    assert "failed unexpectedly (RuntimeError: disk on fire)" in res[0] and res[1] == "error: file not found"


def test_rate_limit_waits_and_malformed_tool_call_is_resampled(fake_api):
    api = fake_api([(429, {"error": {"message": "Rate limit. Please try again in 1.5s"}}, {}),
                    (400, {"error": {"code": "tool_use_failed", "message": "tool_use_failed"}}, {}),
                    text("fine")])
    llm = api.llm()
    r = run(Agent(llm, Registry().add(ADD)))
    assert r.answer == "fine" and llm.usage.waits == [2.0] and api.requests[-1]["temperature"] == 0.3


def test_api_error_ends_the_run_cleanly(fake_api):
    api = fake_api([(401, {"error": {"message": "Invalid API Key"}}, {})])
    r = run(Agent(api.llm(), Registry().add(ADD)))
    assert r.stopped == "error" and "Invalid API Key" in r.answer


def test_think_blocks_are_stripped_and_effort_only_for_gpt_oss(fake_api):
    api = fake_api([text("<think>hmm</think>Answer.")])
    r = run(Agent(api.llm(model="qwen/qwen3.8-27b"), Registry()))
    assert r.answer == "Answer." and "reasoning_effort" not in api.requests[0]


def test_native_citation_marks_are_normalized(fake_api):
    api = fake_api([text("Version 2026-07-28 【1†L13-L22】 is stateless 【2】.")])
    r = run(Agent(api.llm(), Registry()))
    assert r.answer == "Version 2026-07-28 [1] is stateless [2]."


BIG = Tool("page", "Read a page.", {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
           lambda n: f"PAGE {n} " + "lorem ipsum dolor sit amet " * 200)


def test_old_tool_results_are_trimmed_to_fit_the_request_limit(fake_api):
    api = fake_api([calls(("page", {"n": i})) for i in range(1, 5)] + [text("done")])
    r = run(Agent(api.llm(), Registry().add(BIG), context_limit=3500, max_tokens_per_call=500))
    last = api.requests[-1]["messages"]
    tool_msgs = [m["content"] for m in last if m["role"] == "tool"]
    assert r.answer == "done" and len(tool_msgs) == 4
    assert all("trimmed to save context" in m for m in tool_msgs[:2]) and tool_msgs[0].startswith("PAGE 1")
    assert "trimmed" not in tool_msgs[-1]                                       # the newest result stays whole
    assert len(json.dumps(last)) / 3.5 + 500 <= 3500
    assert any(e["type"] == "trim" for e in r.events)


def test_request_too_large_is_retried_smaller(fake_api):
    too_big = (413, {"error": {"message": "Request too large for model on tokens per minute (TPM): Limit 8000, Requested 9042"}}, {})
    api = fake_api([calls(("page", {"n": 1})), calls(("page", {"n": 2})), calls(("page", {"n": 3})), too_big, text("ok")])
    r = run(Agent(api.llm(), Registry().add(BIG)))
    assert r.answer == "ok" and r.stopped == "answer"
    assert len(json.dumps(api.requests[-1]["messages"])) < len(json.dumps(api.requests[-2]["messages"]))


def test_budget_warning_comes_while_the_model_can_still_act(fake_api):
    api = fake_api([calls(("add", {"a": 1, "b": 1}), prompt=800), calls(("add", {"a": 2, "b": 2})), text("saved")])
    r = run(Agent(api.llm(), Registry().add(ADD), token_budget=1000))
    assert "almost used its budget" in api.requests[1]["messages"][-1]["content"] and r.answer == "saved"
    assert sum(e["type"] == "warning" for e in r.events) == 1
