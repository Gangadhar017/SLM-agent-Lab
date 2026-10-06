# Evaluation summary

| model | n | acc | strict acc | acc (iid) | acc (ood) | tool calls/task | tool error rate | malformed rate | compl. tokens/task | tokens/s |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 150 | 31.3% | 30.0% | 36.1% | 12.9% | 0.53 | 35.0% | 0.0% | 87.1 | 3.12 |
| granite-4.0-350m | 150 | 58.0% | 50.0% | 65.5% | 29.0% | 0.99 | 16.1% | 0.7% | 61.4 | 4.42 |


## Accuracy by category

| model | calc_single | convert_single | sql_single | doc_single | multi_step | no_tool | unanswerable |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 20% | 67% | 33% | 0% | 4% | 80% | 40% |
| granite-4.0-350m | 87% | 90% | 87% | 0% | 8% | 30% | 60% |


## Failure taxonomy (count of tasks)

| model | no_final_answer | malformed_tool_call | hallucinated_tool | invalid_arguments | no_tool_call | wrong_tool | incomplete_chain | semantically_wrong_call | wrong_answer_after_correct_tools | unnecessary_tool_then_wrong | fabricated_answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 1 | 0 | 11 | 2 | 56 | 15 | 9 | 1 | 3 | 2 | 3 |
| granite-4.0-350m | 0 | 1 | 0 | 8 | 0 | 24 | 16 | 4 | 1 | 7 | 2 |


## Failure groups

| model | syntax (can't talk to tools) | planning (wrong/no tool) | reasoning (wrong use/answer) |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 14 (9%) | 82 (55%) | 7 (5%) |
| granite-4.0-350m | 9 (6%) | 47 (31%) | 7 (5%) |

