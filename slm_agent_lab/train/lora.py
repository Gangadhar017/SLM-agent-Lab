"""LoRA fine-tuning with assistant-only loss masking, plus a controlled rank x learning-rate x seed sweep.

Design of the controlled experiment: everything except (rank, lr, seed) is fixed -- data, epochs, batch size,
alpha/rank ratio, dropout, target modules, max length, optimiser, schedule. The sweep writes one row per run to
sweep.csv, so plots in the notebook come straight from the data.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class TrainConfig:
    model_id: str
    data_path: str
    out_dir: str
    rank: int = 8
    alpha_ratio: float = 2.0          # alpha = alpha_ratio * rank, kept constant across ranks
    lr: float = 2e-4
    seed: int = 0
    epochs: float = 2.0
    batch_size: int = 4
    grad_accum: int = 1
    max_len: int = 1024
    dropout: float = 0.05
    target_modules: tuple[str, ...] = tuple(DEFAULT_TARGETS)
    warmup_ratio: float = 0.05
    max_examples: int | None = None
    eval_fraction: float = 0.1
    dtype: str = "float32"
    log_every: int = 10


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(path: str | Path, max_examples: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        ex = [json.loads(line) for line in f if line.strip()]
    return ex[:max_examples] if max_examples else ex


def tokenize_with_assistant_mask(tokenizer, messages: list[dict], max_len: int) -> dict:
    """Render the whole conversation once, then label only the tokens of assistant turns (tool calls + final
    answer). Everything else (system, user, tool observations) is masked with -100."""
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_len)
    offsets = enc["offset_mapping"]
    labels = [-100] * len(enc["input_ids"])
    # assistant spans = text between render(messages[:i] + generation prompt) and render(messages[:i+1])
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        upto = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False)
        start, end = len(prefix), len(upto)
        if not upto.startswith(prefix[: min(len(prefix), 50)]):
            start = full.find(m.get("content") or "", 0)  # fallback: locate content directly
        for t, (a, b) in enumerate(offsets):
            if a >= start and b <= end and b > a:
                labels[t] = enc["input_ids"][t]
    return {"input_ids": enc["input_ids"], "labels": labels}


def _collate(batch: list[dict], pad_id: int):
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        attn[i, :L] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def _eval_loss(model, batches) -> float:
    import torch
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for b in batches:
            out = model(**{k: v.to(model.device) for k, v in b.items()})
            total += float(out.loss) * int((b["labels"] != -100).sum())
            n += int((b["labels"] != -100).sum())
    model.train()
    return total / max(1, n)


def train_lora(cfg: TrainConfig, progress: bool = True) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.config.use_cache = False

    lcfg = LoraConfig(r=cfg.rank, lora_alpha=int(cfg.alpha_ratio * cfg.rank), lora_dropout=cfg.dropout,
                      target_modules=list(cfg.target_modules), task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    examples = load_examples(cfg.data_path, cfg.max_examples)
    rng = random.Random(cfg.seed)
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * cfg.eval_fraction))
    eval_ex, train_ex = examples[:n_eval], examples[n_eval:]
    train_tok = [tokenize_with_assistant_mask(tokenizer, e["messages"], cfg.max_len) for e in train_ex]
    eval_tok = [tokenize_with_assistant_mask(tokenizer, e["messages"], cfg.max_len) for e in eval_ex]
    train_tok = [t for t in train_tok if any(l != -100 for l in t["labels"])]
    eval_batches = [_collate(eval_tok[i:i + cfg.batch_size], tokenizer.pad_token_id)
                    for i in range(0, len(eval_tok), cfg.batch_size)]

    steps_per_epoch = math.ceil(len(train_tok) / (cfg.batch_size * cfg.grad_accum))
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)

    model.train()
    log: list[dict] = []
    step, t0 = 0, time.perf_counter()
    eval_before = _eval_loss(model, eval_batches)
    while step < total_steps:
        order = list(range(len(train_tok)))
        rng.shuffle(order)
        for i in range(0, len(order), cfg.batch_size * cfg.grad_accum):
            if step >= total_steps:
                break
            chunk = order[i:i + cfg.batch_size * cfg.grad_accum]
            loss_acc = 0.0
            for j in range(0, len(chunk), cfg.batch_size):
                batch = _collate([train_tok[k] for k in chunk[j:j + cfg.batch_size]], tokenizer.pad_token_id)
                out = model(**{k: v.to(device) for k, v in batch.items()})
                (out.loss / cfg.grad_accum).backward()
                loss_acc += float(out.loss) / cfg.grad_accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == total_steps:
                rec = {"step": step, "train_loss": round(loss_acc, 4), "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.perf_counter() - t0, 1)}
                log.append(rec)
                if progress:
                    print(f"  step {step}/{total_steps}  loss {loss_acc:.4f}  ({rec['elapsed_s']}s)", flush=True)
    eval_after = _eval_loss(model, eval_batches)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    result = {
        **asdict(cfg),
        "target_modules": ",".join(cfg.target_modules),
        "n_train": len(train_tok), "n_eval": len(eval_tok), "total_steps": total_steps,
        "trainable_params": n_trainable, "total_params": n_total,
        "trainable_pct": round(100 * n_trainable / n_total, 4),
        "eval_loss_before": round(eval_before, 4), "eval_loss_after": round(eval_after, 4),
        "final_train_loss": log[-1]["train_loss"] if log else None,
        "train_time_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "train_result.json").write_text(json.dumps({"result": result, "log": log}, indent=2), encoding="utf-8")
    return result


def append_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
