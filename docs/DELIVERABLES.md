# Deliverables and how to talk about them

This file is for the author: what exists, what each piece demonstrates, and what to say about it in an
application or interview. The README is for readers of the repo.

## What is delivered

| # | deliverable | where | demonstrates |
|---|---|---|---|
| 1 | Tool-calling agent harness with 4 deterministic tools, 5-dialect tool-call parser, resumable trajectory logger | `slm_agent_lab/{tools,agent,eval}` | agentic systems with small LMs; tool use / function calling |
| 2 | 600-task benchmark (300 test / 300 train, 7 categories, 56 OOD test tasks), generated from seeded templates with exact ground truth | `data/tasks/*.jsonl`, `slm_agent_lab/tasks/generate.py` | designing a controlled evaluation |
| 3 | 13-label failure taxonomy with a fixed decision tree, incl. the "did the agent *have* the answer?" check | `slm_agent_lab/eval/taxonomy.py` | testing and debugging of agents |
| 4 | Baseline results for IBM Granite 4.0 350M and Qwen2.5 0.5B with figures, tables and per-task logs | `results/runs/`, `results/report/`, `docs/figures/` | ability to run and report an experiment |
| 5 | Distillation data builder (teacher-filtered and oracle), LoRA SFT with assistant-only masking, controlled rank × lr × seed sweep | `slm_agent_lab/train/`, `scripts/lora_sweep.py`, `notebooks/02_*.ipynb` | PyTorch, LoRA/PEFT, knowledge distillation, controlled experiments |
| 6 | Reproduction plan + code for LoRA paper §7 (rank sufficiency, subspace similarity with random baseline) | `REPRODUCTION.md`, `slm_agent_lab/train/subspace.py` | reading papers and reproducing results |
| 7 | Serving benchmark (throughput / TTFT / latency / accuracy-through-the-server) + vLLM on Kubernetes with Prometheus | `serve/` | AI accelerator software stack, Triton/vLLM, K8s, observability |
| 8 | 17 unit tests, notebooks, README, MIT licence | `tests/`, `notebooks/`, `README.md` | engineering hygiene |

## What is finished vs. what is scaffolded (be precise about this)

* **Finished and run**: stages 1 (harness, tasks, taxonomy, two baselines on 150 tasks, figures) and the CPU
  smoke test of stage 2/3 code.
* **Code complete, not yet run at full scale**: stage 2 sweep and the stage 3 numbers (need a GPU; the Kaggle
  notebook is the runbook, ≈ 2 h). Say "the pipeline is verified end to end; the sweep is the next step".
* **Written, not executed**: stage 4 (needs a GPU node). Say so.

## One-paragraph description

> I built an evaluation harness for small tool-calling language-model agents (IBM Granite 4.0 350M/1B,
> Qwen2.5 0.5B) with four deterministic tools, a 300-task benchmark with exact ground truth and held-out OOD
> templates, full trajectory logging and a 13-label failure taxonomy that separates syntax failures (malformed
> or hallucinated calls) from planning failures (wrong/no tool) and reasoning failures (right tool, wrong use or
> wrong synthesis). On top of it I set up rejection-sampled distillation from a 3B Granite teacher into the 350M
> student with LoRA, a controlled rank × learning-rate × seed sweep, and a reproduction of the LoRA paper's
> subspace-similarity analysis on my own adapters; the serving stage benchmarks the trained models on vLLM
> under fp16/int8/int4 and runs the same harness through the server to see which failure type quantisation
> triggers first.

## Likely interview questions and honest answers

* *Why deterministic tools and synthetic tasks?* Exact ground truth; failures are attributable to the model, not
  the environment. The cost is realism, which is noted as a limitation.
* *Why is `expected_tools` not part of the score?* There can be several valid tool paths (one SQL vs SQL +
  calculator). Correctness is answer-based; tool paths are diagnostic only.
* *What did you find?* Granite-4.0-350M 58 % vs Qwen2.5-0.5B 31 % on the same 150 tasks. Granite's tool-call
  syntax is essentially perfect (0.7 % malformed) and its failures are planning failures: it never picks
  `doc_search` for a plain policy question (0/20, routes to `sql_query`) unless the prompt names a document, and
  it stops multi-step chains after the first tool. Qwen's dominant failure is answering from memory without any
  tool call (56/150) and hallucinating tool names from the calculator description (`sqrt`, `div`). Both corrupt
  numbers when copying them from observations (sign dropped, digits transposed). Full numbers: README "Key
  findings" and `results/report/summary.md`.
* *What would you do with more compute?* The full 54-run grid, 1B/1.5B students, DPO from the logged
  correct/incorrect pairs, int4 serving sweep, Triton kernel.

## Commands to regenerate everything

```bash
python -m pytest -q
python scripts/make_tasks.py
python scripts/run_eval.py --model ibm-granite/granite-4.0-350m --limit 150
python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 150
python scripts/make_report.py
python scripts/build_sft_data.py --source oracle
python scripts/lora_sweep.py --model ibm-granite/granite-4.0-350m --ranks 4,8 --lrs 2e-4 --seeds 0 --epochs 0.4 --max-examples 24 --batch-size 1 --grad-accum 4 --max-len 512 --out results/sweep_smoke
python scripts/subspace_analysis.py results/sweep_smoke/r4_lr0.0002_s0 results/sweep_smoke/r8_lr0.0002_s0
```
