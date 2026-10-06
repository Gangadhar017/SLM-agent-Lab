"""The agent loop: render -> generate -> parse -> execute -> observe, with full trajectory logging."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from ..tasks.schema import Task
from ..tools.registry import TOOL_SCHEMAS, execute_tool
from .parsing import parse_tool_calls

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using the available tools.\n"
    "- Use sql_query for questions about employees, orders or products in the company database.\n"
    "- Use doc_search for questions about company policies, product specifications, logistics or reports.\n"
    "- Use calculator for arithmetic and unit_convert for unit conversions.\n"
    "- Answer general-knowledge questions directly without tools.\n"
    "- If the information needed is not available in the tools' results, say so clearly instead of guessing.\n"
    "Call one tool at a time and wait for its result. When you have the answer, reply with one short sentence "
    "that ends with 'Final answer: <value>'."
)

MALFORMED_FEEDBACK = (
    "Your tool call could not be parsed. Reply with exactly one valid call in the form "
    '<tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>, or give the final answer.'
)


@dataclass
class Step:
    index: int
    raw_output: str
    parsed_calls: list[dict] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: str = "eos"
    text_after_calls: str = ""


@dataclass
class Trajectory:
    task_id: str
    category: str
    model: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    terminated_by: str = ""          # final_answer | max_steps | loop_detected | malformed | backend_error
    n_steps: int = 0
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    tools_used: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_time_s: float = 0.0
    wall_time_s: float = 0.0
    had_malformed: bool = False
    n_repaired_json: int = 0
    call_sources: list[str] = field(default_factory=list)
    fabricated_text_after_call: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _call_key(name: str, arguments: dict) -> str:
    return name + ":" + json.dumps(arguments, sort_keys=True, default=str)


def run_task(backend, task: Task, max_steps: int = 6, max_calls_per_step: int = 3, tools: list[dict] | None = None,
             system_prompt: str = SYSTEM_PROMPT) -> Trajectory:
    tools = TOOL_SCHEMAS if tools is None else tools
    traj = Trajectory(task_id=task.id, category=task.category, model=backend.label)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task.prompt}]
    seen_calls: set[str] = set()
    malformed_retries = 0
    t_start = time.perf_counter()

    try:
        for step_idx in range(max_steps):
            prompt_text = backend.render(messages, tools)
            gen = backend.generate(prompt_text)
            parsed = parse_tool_calls(gen.text)
            step = Step(index=step_idx, raw_output=gen.text, parsed_calls=[c.to_dict() for c in parsed.calls],
                        malformed=parsed.malformed, prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens, latency_s=round(gen.latency_s, 3),
                        stop_reason=gen.stop_reason, text_after_calls=parsed.text_after_calls[:500])
            traj.prompt_tokens += gen.prompt_tokens
            traj.completion_tokens += gen.completion_tokens
            traj.generation_time_s += gen.latency_s
            if parsed.malformed:
                traj.had_malformed = True
            traj.n_repaired_json += sum(1 for c in parsed.calls if c.repaired)
            if len(parsed.text_after_calls) > 40:
                traj.fabricated_text_after_call = True

            if parsed.calls:
                calls = parsed.calls[:max_calls_per_step]
                loop_hit = False
                results = []
                for c in calls:
                    key = _call_key(c.name, c.arguments)
                    if key in seen_calls:
                        loop_hit = True
                    seen_calls.add(key)
                    res = execute_tool(c.name, c.arguments)
                    results.append(res)
                    traj.n_tool_calls += 1
                    traj.n_tool_errors += 0 if res.ok else 1
                    traj.call_sources.append(c.source)
                    if c.name not in traj.tools_used:
                        traj.tools_used.append(c.name)
                step.tool_results = [r.to_dict() for r in results]
                traj.steps.append(step)
                if loop_hit:
                    traj.terminated_by = "loop_detected"
                    traj.final_answer = parsed.final_text
                    break
                messages.append({
                    "role": "assistant",
                    "content": parsed.final_text or "",
                    "tool_calls": [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in calls],
                })
                for c, r in zip(calls, results):
                    messages.append({"role": "tool", "name": c.name, "content": r.observation()})
                continue

            traj.steps.append(step)
            if parsed.malformed and malformed_retries < 1:
                malformed_retries += 1
                messages.append({"role": "assistant", "content": gen.text})
                messages.append({"role": "user", "content": MALFORMED_FEEDBACK})
                continue

            traj.final_answer = parsed.final_text or gen.text
            traj.terminated_by = "malformed" if parsed.malformed else "final_answer"
            break
        else:
            traj.terminated_by = "max_steps"
    except Exception as exc:  # backend failure: keep the partial trajectory, mark it, carry on with the run
        traj.terminated_by = "backend_error"
        traj.error = f"{type(exc).__name__}: {exc}"

    traj.n_steps = len(traj.steps)
    traj.wall_time_s = round(time.perf_counter() - t_start, 3)
    return traj
