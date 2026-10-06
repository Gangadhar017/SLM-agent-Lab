# Reproduction: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021), §7

## What the paper claims

1. **A very low rank is enough.** Adapting GPT-3 175B with LoRA at r = 1–4 performs on par with r = 64 on
   WikiSQL and MultiNLI (Table 6 in the paper): the update ΔW has a small "intrinsic rank".
2. **The learned subspaces overlap.** For adapters trained at r = 8 and r = 64 on the same data, the top
   singular directions of A_r=8 are contained in those of A_r=64: the normalised subspace similarity
   φ(A_r=8, A_r=64, i, j) = ‖U_8^{i⊤} U_64^{j}‖²_F / min(i, j) is high (> 0.5) for the top direction(s) and
   drops towards the random-matrix baseline for the remaining directions (Figure 3). Different random seeds at
   the same rank also agree on the top directions (Figure 4).

## What this repo reproduces

| paper artefact | our artefact | script |
|---|---|---|
| Table 6: accuracy vs. rank r ∈ {1, 2, 4, 8, 64} | `results/sweep/<model>/sweep.csv` + `docs/figures/lora_rank_sweep.png`: task accuracy vs. r ∈ {1, 2, 4, 8, 16, 64}, mean ± std over 3 seeds, for 3 learning rates | `scripts/lora_sweep.py` |
| Figure 3: φ(A_r=8, A_r=64) heat-map | `docs/figures/subspace_r4_vs_r64_q.png` (+ random-matrix baseline) | `scripts/subspace_analysis.py` |
| Figure 4: φ between two seeds at the same rank | `docs/figures/subspace_r64_seed0_vs_seed1_q.png` | `scripts/subspace_analysis.py` |

The exact formula from the paper is implemented in `slm_agent_lab/train/subspace.py` (`phi`, `phi_grid`) and
averaged over all adapted modules of one type (e.g. every `q_proj`). A random-Gaussian baseline with the same
shapes is computed alongside so "high" has a reference.

## Deliberate differences from the paper's setup (and why)

| | paper | here | why it matters |
|---|---|---|---|
| model | GPT-3 175B | Granite 4.0 350M (and 1B) | at 350M the intrinsic rank of the task may be *higher*: fewer parameters, less redundancy. If low rank still suffices, the claim is stronger; if not, we learn where it breaks. |
| task | WikiSQL (NL→SQL), MultiNLI | tool-calling SFT: emit a JSON tool call + read observation + answer | a structured-output task; closer to the lab's "agentic SLM" interest |
| data size | 56k / 393k examples | ≈ 290 trajectories (teacher-filtered or oracle) | tiny data favours low rank; also why we hold out OOD templates and documents |
| adapted weights | W_q, W_v | W_q, W_k, W_v, W_o | PEFT default for these architectures; alpha/r kept constant (2) across ranks as in the paper's practice |
| metric | task accuracy | harness accuracy on the held-out split (same tolerance rules as all other experiments) + eval loss | accuracy is the quantity that matters for an agent; loss is reported for completeness |
| seeds | 1 | 3 per cell | error bars; the reduced grid is 24 runs |

## Protocol (controlled)

Vary only **rank**, **learning rate**, **seed**. Fixed: data file and order, 2 epochs, batch 4, grad-accum 1,
dropout 0.05, alpha = 2·r, targets q/k/v/o, max length 1024, AdamW (wd 0), linear schedule, 5 % warm-up,
assistant-only loss masking, greedy decoding at evaluation, the same 100 held-out tasks for every run.

```bash
python scripts/make_tasks.py
python scripts/build_sft_data.py --source oracle            # or --source teacher --run results/runs/<teacher_train>
python scripts/lora_sweep.py --model ibm-granite/granite-4.0-350m --ranks 1,2,4,8,16,64 --lrs 1e-4,2e-4,5e-4 --seeds 0,1,2 --eval-tasks 100
python scripts/subspace_analysis.py results/sweep/granite-4.0-350m/r8_lr0.0002_s0 results/sweep/granite-4.0-350m/r64_lr0.0002_s0 --module q_proj
```

## Results

**Status: pipeline verified end-to-end on CPU; the full sweep is a GPU job (notebook `02_lora_sweep_kaggle.ipynb`,
≈ 2 h on a free T4).** The CPU smoke run (`results/sweep_smoke/`: r = 4 and 8, 40 examples, 2 optimiser
steps) exists only to prove the training, masking, adapter saving and subspace code run; its numbers are not
results and are not reported here.

Fill in after the sweep:

| rank r | trainable params (%) | accuracy lr=1e-4 | accuracy lr=2e-4 | accuracy lr=5e-4 |
|---|---|---|---|---|
| baseline (no LoRA) | 0 | — | — | — |
| 1 | | | | |
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |
| 16 | | | | |
| 64 | | | | |

| comparison | φ top-1 | φ diag (first 4) | random baseline top-1 |
|---|---|---|---|
| r=4 vs r=64 (q_proj) | | | |
| r=64 seed 0 vs seed 1 (q_proj) | | | |

## Where we expect numbers to differ, and how to read them

* **Rank curve.** The paper sees a flat curve from r = 1. With a 350M model and a structured-output task we
  expect a rise from r = 1 to r ≈ 4–8 and a plateau after: the JSON dialect plus argument formatting is more
  than a one-direction change for a small model. A plateau anywhere below 16 supports the paper's conclusion
  at this scale; a monotonic rise to 64 would contradict it and would be the most interesting outcome.
* **Learning rate interaction.** Higher lr can mask rank effects (the paper fixed lr per task). Reporting the
  full grid, not the best cell, is the point of the controlled design.
* **Subspace similarity.** The absolute φ depends on d (hidden size 1024 here vs 12288), so compare against
  the random baseline computed for *our* shapes rather than against the paper's colour scale.
* **Noise.** 100 evaluation tasks give a ±5 pp 95 % interval per run; differences smaller than that between
  ranks are not claims.
