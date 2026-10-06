import pytest

from slm_agent_lab.tools import execute_tool
from slm_agent_lab.tools.calculator import CalculatorError, calculate
from slm_agent_lab.tools.doc_search import search
from slm_agent_lab.tools.sql_query import SQLError, run_query, scalar
from slm_agent_lab.tools.unit_convert import UnitError, convert


def test_calculator_basic():
    assert calculate("17.5 / 100 * 8432")["result"] == pytest.approx(1475.6)
    assert calculate("(12 + 8) * 3 - 4")["result"] == 56
    assert calculate("sqrt(144)")["result"] == 12
    assert calculate("8,432 * 2")["result"] == 16864  # thousands separator tolerated
    assert calculate("2^10")["result"] == 1024


def test_calculator_rejects_code():
    with pytest.raises(CalculatorError):
        calculate("__import__('os').system('echo hi')")
    with pytest.raises(CalculatorError):
        calculate("2 ** 100000")
    with pytest.raises(CalculatorError):
        calculate("1 / 0")


def test_unit_convert():
    assert convert(1, "mi", "km")["value"] == pytest.approx(1.609344)
    assert convert(100, "f", "c")["value"] == pytest.approx(37.777778, abs=1e-5)
    assert convert(2, "pounds", "kilograms")["value"] == pytest.approx(0.907185, abs=1e-5)
    assert convert("3", "hours", "minutes")["value"] == 180
    with pytest.raises(UnitError):
        convert(1, "kg", "km")
    with pytest.raises(UnitError):
        convert(1, "parsec", "km")


def test_sql_read_only_and_deterministic():
    assert scalar("SELECT COUNT(*) FROM employees") == 60
    assert scalar("SELECT COUNT(*) FROM orders") == 120
    assert scalar("SELECT COUNT(*) FROM products") == 12
    with pytest.raises(SQLError):
        run_query("DROP TABLE employees")
    with pytest.raises(SQLError):
        run_query("SELECT 1; DELETE FROM employees")
    with pytest.raises(SQLError):
        run_query("UPDATE employees SET salary = 0")
    assert scalar("SELECT COUNT(*) FROM employees") == 60


def test_doc_search_finds_the_right_document():
    res = search("travel policy meal per diem", k=2)
    assert res["results"][0]["id"] == "travel-policy"
    res = search("Laptop Pro 14 weight", k=1)
    assert res["results"][0]["id"] == "laptop-pro-14-spec"


def test_execute_tool_dispatch_and_errors():
    assert execute_tool("calculator", {"expression": "2+2"}).output["result"] == 4
    r = execute_tool("calculator", {"expr": "2+2"})  # alias accepted and noted
    assert r.ok and "renamed argument expr->expression" in r.output["_notes"]
    r = execute_tool("unit_convert", {"value": "12", "from_unit": "km", "to_unit": "mi"})
    assert r.ok and r.output["value"] == pytest.approx(7.456454, abs=1e-5)
    assert execute_tool("teleport", {}).error_kind == "unknown_tool"
    assert execute_tool("unit_convert", {"value": 1}).error_kind == "missing_argument"
    assert execute_tool("sql_query", {"query": "SELECT * FROM nope"}).error_kind == "execution"
    assert execute_tool("calculator", {"expression": "17.5% of 8432"}).error_kind == "execution"
