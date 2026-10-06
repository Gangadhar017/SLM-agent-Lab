from slm_agent_lab.eval.scoring import score_answer
from slm_agent_lab.eval.taxonomy import classify
from slm_agent_lab.tasks.generate import generate_tasks, task_counts


def test_number_scoring_strict_vs_lenient():
    exp = {"type": "number", "value": 1475.6, "abs_tol": 0.011}
    assert score_answer(exp, "Final answer: 1,475.60")["strict"]
    assert score_answer(exp, "17.5% of 8432 is 1475.6 USD.")["strict"]
    r = score_answer(exp, "1475.6 is the value, which is about 1.48k. Final answer: 1.48")
    assert r["correct"] and not r["strict"]
    assert not score_answer(exp, "Final answer: 1480")["correct"]


def test_string_and_unanswerable_scoring():
    assert score_answer({"type": "string", "any_of": ["au"]}, "The symbol is Au.")["correct"]
    assert not score_answer({"type": "string", "any_of": ["au"]}, "because of audio")["correct"]
    assert score_answer({"type": "string", "any_of": ["structured query language"]}, "SQL = Structured Query Language")["correct"]
    assert score_answer({"type": "unanswerable"}, "There is no employee named Zubin Mistry in the database.")["correct"]
    assert not score_answer({"type": "unanswerable"}, "Final answer: 85000")["correct"]


def _traj(**kw):
    base = {"steps": [], "final_answer": "Final answer: 1", "terminated_by": "final_answer", "tools_used": [],
            "n_tool_calls": 0, "had_malformed": False, "n_repaired_json": 0, "fabricated_text_after_call": False}
    base.update(kw)
    return base


def _task(category="sql_single", tools=("sql_query",), expected=None):
    return {"category": category, "expected_tools": list(tools), "gold_calls": [{} for _ in tools],
            "expected": expected or {"type": "number", "value": 7, "abs_tol": 0.5}}


def _step(name, ok=True, output=None, error_kind=None):
    return {"tool_results": [{"name": name, "ok": ok, "output": output, "error_kind": error_kind, "arguments": {}}]}


def test_taxonomy_decision_tree():
    wrong = {"correct": False, "strict": False}
    assert classify(_task(), _traj(), wrong)["label"] == "no_tool_call"
    assert classify(_task(), _traj(had_malformed=True, terminated_by="malformed"), wrong)["label"] == "malformed_tool_call"
    assert classify(_task(), _traj(tools_used=["teleport"], n_tool_calls=1,
                                   steps=[_step("teleport", ok=False, error_kind="unknown_tool")]), wrong)["label"] == "hallucinated_tool"
    assert classify(_task(), _traj(tools_used=["sql_query"], n_tool_calls=1,
                                   steps=[_step("sql_query", ok=False, error_kind="execution")]), wrong)["label"] == "invalid_arguments"
    assert classify(_task(), _traj(tools_used=["calculator"], n_tool_calls=1,
                                   steps=[_step("calculator", output={"result": 3})]), wrong)["label"] == "wrong_tool"
    # planning error beats execution error: wrong tool *and* a rejected call is still a wrong-tool failure
    assert classify(_task("doc_single", ("doc_search",)), _traj(tools_used=["sql_query"], n_tool_calls=1,
                    steps=[_step("sql_query", ok=False, error_kind="execution")]), wrong)["label"] == "wrong_tool"
    assert classify(_task(), _traj(tools_used=["sql_query"], n_tool_calls=1,
                                   steps=[_step("sql_query", output={"rows": [[7]]})]), wrong)["label"] == "wrong_answer_after_correct_tools"
    assert classify(_task(), _traj(tools_used=["sql_query"], n_tool_calls=1,
                                   steps=[_step("sql_query", output={"rows": [[3]]})]), wrong)["label"] == "semantically_wrong_call"
    multi = _task("multi_step", ("doc_search", "calculator"))
    assert classify(multi, _traj(tools_used=["doc_search"], n_tool_calls=1,
                                 steps=[_step("doc_search", output={"text": "nothing"})]), wrong)["label"] == "incomplete_chain"
    assert classify(_task("no_tool", ()), _traj(), wrong)["label"] == "knowledge_error"
    assert classify(_task("unanswerable", ("sql_query",), {"type": "unanswerable"}), _traj(), wrong)["label"] == "fabricated_answer"
    assert classify(_task(), _traj(terminated_by="max_steps", final_answer=""), wrong)["label"] == "no_final_answer"
    ok = {"correct": True, "strict": True}
    assert classify(_task(), _traj(), ok)["label"] == "success"


def test_task_generation_is_deterministic_and_disjoint():
    a, b = generate_tasks(seed=0, split="test"), generate_tasks(seed=0, split="test")
    assert [t.id for t in a] == [t.id for t in b]
    train = generate_tasks(seed=1, split="train", exclude_prompts={t.prompt for t in a})
    assert not ({t.prompt for t in a} & {t.prompt for t in train})
    assert len(train) >= 280
    assert not any(t.ood for t in train)
    assert any(t.ood for t in a)
    counts = task_counts(a)
    assert counts["calc_single"] == 60 and counts["no_tool"] == 20
    assert len(a) == 300
