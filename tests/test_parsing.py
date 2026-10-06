from slm_agent_lab.agent.parsing import parse_tool_calls


def test_hermes_tool_call_block():
    text = 'Let me check.\n<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call>'
    r = parse_tool_calls(text)
    assert len(r.calls) == 1
    assert r.calls[0].name == "calculator"
    assert r.calls[0].arguments == {"expression": "2+2"}
    assert r.calls[0].source == "tool_call_tag"
    assert r.final_text == "Let me check."
    assert not r.malformed


def test_unterminated_block_and_list():
    text = '<tool_call>[{"name": "doc_search", "arguments": {"query": "travel policy"}}, {"name": "calculator", "arguments": {"expression": "45*3"}}]'
    r = parse_tool_calls(text)
    assert [c.name for c in r.calls] == ["doc_search", "calculator"]


def test_granite3_tag():
    text = '<|tool_call|>[{"name": "sql_query", "arguments": {"query": "SELECT 1"}}]'
    r = parse_tool_calls(text)
    assert r.calls and r.calls[0].source == "granite_tag"


def test_fenced_and_bare_json():
    text = '```json\n{"name": "unit_convert", "arguments": {"value": 3, "from_unit": "mi", "to_unit": "km"}}\n```'
    assert parse_tool_calls(text).calls[0].source == "fenced"
    text = 'I will call {"name": "sql_query", "parameters": {"query": "SELECT COUNT(*) FROM employees"}} now'
    r = parse_tool_calls(text)
    assert r.calls[0].source == "bare_json" and r.calls[0].arguments == {"query": "SELECT COUNT(*) FROM employees"}


def test_python_style_and_single_quotes_repair():
    r = parse_tool_calls('calculator(expression="3*4")')
    assert r.calls[0].source == "python_call" and r.calls[0].arguments == {"expression": "3*4"}
    r = parse_tool_calls("<tool_call>{'name': 'calculator', 'arguments': {'expression': '3*4'}}</tool_call>")
    assert r.calls[0].repaired is True
    r = parse_tool_calls('<tool_call>{"name": "calculator", "arguments": "{\\"expression\\": \\"5*5\\"}"}</tool_call>')
    assert r.calls[0].arguments == {"expression": "5*5"}


def test_malformed_is_reported_not_swallowed():
    r = parse_tool_calls('<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"</tool_call>')
    assert not r.calls and r.malformed
    r = parse_tool_calls("The answer is 42. Final answer: 42")
    assert not r.calls and not r.malformed and r.final_text.endswith("42")


def test_fabricated_observation_is_visible():
    text = ('<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>\n'
            '<tool_response>{"result": 4}</tool_response>\nFinal answer: 4')
    r = parse_tool_calls(text)
    assert r.calls and "tool_response" in r.text_after_calls
