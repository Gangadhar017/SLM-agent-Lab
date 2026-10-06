| variant | n | accuracy | doc_search used | doc_search first call | top failure labels |
|---|---|---|---|---|---|
| baseline | 20 | 0% | 0% | 0% | wrong_tool 19, malformed_tool_call 1 |
| doc_tool_first | 20 | 5% | 0% | 0% | wrong_tool 18, malformed_tool_call 1 |
| cue_prefix | 20 | 75% | 95% | 95% | wrong_answer_after_correct_tools 3, wrong_tool 1, semantically_wrong_call 1 |
| sql_removed | 20 | 50% | 50% | 50% | wrong_tool 9, malformed_tool_call 1 |
| doc_tool_only | 20 | 75% | 90% | 90% | semantically_wrong_call 2, no_tool_call 2, wrong_answer_after_correct_tools 1 |
