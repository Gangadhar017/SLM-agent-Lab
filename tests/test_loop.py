"""Agent-loop tests with a scripted fake backend: no model download, runs in milliseconds.

These pin down the control flow that the failure taxonomy depends on: observation threading, loop detection,
malformed-call retry, step limits and backend failures.
"""

from slm_agent_lab.agent.backends import GenOutput
from slm_agent_lab.agent.loop import MALFORMED_FEEDBACK, run_task
from slm_agent_lab.eval.scoring import score_answer
from slm_agent_lab.eval.taxonomy import classify
from slm_agent_lab.tasks.schema import Task


class FakeBackend:
    label = "fake"
    model_id = "fake"
    adapter = None
    max_new_tokens = 64

    def __init__(self, outputs, raise_on=None):
        self.outputs = list(outputs)
        self.rendered = []          # messages seen at each render call
        self.raise_on = raise_on

    def render(self, messages, tools):
        self.rendered.append([dict(m) for m in messages])
        return "PROMPT"

    def generate(self, prompt_text):
        if self.raise_on is not None and len(self.rendered) - 1 == self.raise_on:
            raise RuntimeError("boom")
        text = self.outputs.pop(0) if self.outputs else "Final answer: nothing"
        return GenOutput(text=text, prompt_tokens=10, completion_tokens=5, latency_s=0.01, stop_reason="eos")


def _task():
    return Task(id="t1", category="calc_single", template="calc", prompt="What is 6 * 7?",
                expected={"type": "number", "value": 42, "abs_tol": 0.5}, expected_tools=["calculator"],
                gold_calls=[{"name": "calculator", "arguments": {"expression": "6*7"}}], gold_answer="Final answer: 42",
                difficulty=1)


CALL = '<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>'


def test_tool_call_then_answer():
    b = FakeBackend([CALL, "The product is 42. Final answer: 42"])
    traj = run_task(b, _task())
    assert traj.n_steps == 2 and traj.n_tool_calls == 1 and traj.tools_used == ["calculator"]
    assert traj.terminated_by == "final_answer"
    # the second render must contain the assistant tool call and the tool observation
    roles = [m["role"] for m in b.rendered[1]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert '"result": 42' in b.rendered[1][3]["content"]
    score = score_answer(_task().expected, traj.final_answer)
    assert score["correct"] and classify(_task().to_dict(), traj.to_dict(), score)["label"] == "success"


def test_loop_detection():
    b = FakeBackend([CALL, CALL, CALL])
    traj = run_task(b, _task())
    assert traj.terminated_by == "loop_detected" and traj.n_tool_calls == 2
    label = classify(_task().to_dict(), traj.to_dict(), score_answer(_task().expected, traj.final_answer))["label"]
    assert label == "no_final_answer"


def test_malformed_retry_then_recover():
    bad = '<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"</tool_call>'
    b = FakeBackend([bad, CALL, "Final answer: 42"])
    traj = run_task(b, _task())
    assert traj.had_malformed and traj.n_tool_calls == 1 and traj.terminated_by == "final_answer"
    assert b.rendered[1][-1]["content"] == MALFORMED_FEEDBACK  # feedback was given exactly once
    assert score_answer(_task().expected, traj.final_answer)["correct"]


def test_malformed_twice_terminates():
    bad = '<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"</tool_call>'
    traj = run_task(FakeBackend([bad, bad]), _task())
    assert traj.terminated_by == "malformed" and traj.n_tool_calls == 0
    label = classify(_task().to_dict(), traj.to_dict(), score_answer(_task().expected, traj.final_answer))["label"]
    assert label == "malformed_tool_call"


def test_max_steps():
    calls = [f'<tool_call>{{"name": "calculator", "arguments": {{"expression": "{i}+1"}}}}</tool_call>' for i in range(10)]
    traj = run_task(FakeBackend(calls), _task(), max_steps=3)
    assert traj.terminated_by == "max_steps" and traj.n_steps == 3


def test_backend_error_is_recorded_not_raised():
    traj = run_task(FakeBackend([CALL, "Final answer: 42"], raise_on=1), _task())
    assert traj.terminated_by == "backend_error" and "boom" in traj.error and traj.n_steps == 1


def test_hallucinated_tool_feedback_reaches_model():
    b = FakeBackend(['<tool_call>{"name": "sqrt", "arguments": {"value": 9}}</tool_call>', "Final answer: 3"])
    traj = run_task(b, _task())
    assert traj.n_tool_errors == 1
    assert "unknown tool" in b.rendered[1][3]["content"]
