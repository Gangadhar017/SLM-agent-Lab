"""Stage 4 benchmark: throughput / latency / accuracy of a served model across concurrency levels.

Works against any OpenAI-compatible server (vLLM, TGI, llama.cpp). Prompts are the *real agent prompts* (system
prompt + tool schemas + task), rendered with the model's own chat template, so numbers reflect agent workloads
(long prompts, short completions) rather than generic text generation.

  # throughput/latency sweep
  python serve/bench.py --base-url http://localhost:8000 --model ibm-granite/granite-4.0-1b --concurrency 1,4,16,64 --tag fp16
  # same, and also run the task harness against the server (accuracy under this quantization)
  python serve/bench.py --base-url http://localhost:8000 --model ibm-granite/granite-4.0-1b-AWQ --tag int4 --harness 150
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from slm_agent_lab.agent.backends import ChatRenderer  # noqa: E402
from slm_agent_lab.agent.loop import SYSTEM_PROMPT  # noqa: E402
from slm_agent_lab.paths import RESULTS_DIR, TASKS_DIR  # noqa: E402
from slm_agent_lab.tasks.schema import load_tasks  # noqa: E402
from slm_agent_lab.tools.registry import TOOL_SCHEMAS  # noqa: E402


def one_request(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft, n_chunks, usage = None, 0, None
    with requests.post(f"{base_url}/v1/completions", json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices") and obj["choices"][0].get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_chunks += 1
    latency = time.perf_counter() - t0
    tokens = usage["completion_tokens"] if usage else n_chunks
    return {"ttft_s": ttft if ttft is not None else latency, "latency_s": latency, "completion_tokens": tokens}


def run_level(base_url: str, model: str, prompts: list[str], concurrency: int, requests_per_worker: int,
              max_tokens: int, timeout: float) -> dict:
    results: list[dict] = []
    lock = threading.Lock()

    def worker(wid: int) -> None:
        for i in range(requests_per_worker):
            p = prompts[(wid * requests_per_worker + i) % len(prompts)]
            try:
                res = one_request(base_url, model, p, max_tokens, timeout)
            except Exception as exc:  # record failures, keep the benchmark going
                res = {"error": str(exc)}
            with lock:
                results.append(res)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in results if "error" not in r]
    lat = sorted(r["latency_s"] for r in ok) or [0.0]
    ttft = sorted(r["ttft_s"] for r in ok) or [0.0]
    tokens = sum(r["completion_tokens"] for r in ok)
    return {
        "concurrency": concurrency, "requests": len(results), "failed": len(results) - len(ok), "wall_s": round(wall, 2),
        "throughput_tok_s": round(tokens / wall, 2) if wall else 0, "requests_per_s": round(len(ok) / wall, 3) if wall else 0,
        "ttft_p50_s": round(statistics.median(ttft), 3), "ttft_p95_s": round(ttft[int(0.95 * (len(ttft) - 1))], 3),
        "latency_p50_s": round(statistics.median(lat), 3), "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 3),
        "mean_completion_tokens": round(tokens / max(1, len(ok)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--tag", default="fp16", help="configuration label, e.g. fp16 / int8 / int4-awq")
    ap.add_argument("--concurrency", default="1,4,16,64")
    ap.add_argument("--requests-per-worker", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--harness", type=int, default=0, help="also evaluate this many tasks through the server")
    ap.add_argument("--out", default=str(RESULTS_DIR / "serve"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    renderer = ChatRenderer(AutoTokenizer.from_pretrained(args.tokenizer or args.model))
    tasks = load_tasks(TASKS_DIR / "test.jsonl")
    prompts = [renderer.render([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t.prompt}],
                               TOOL_SCHEMAS) for t in tasks[:64]]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bench.csv"
    rows = []
    for c in [int(x) for x in args.concurrency.split(",")]:
        print(f"concurrency={c} ...", flush=True)
        row = {"tag": args.tag, "model": args.model, **run_level(args.base_url, args.model, prompts, c,
                                                                 args.requests_per_worker, args.max_tokens, args.timeout)}
        print(json.dumps(row))
        rows.append(row)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)

    if args.harness:
        from slm_agent_lab.agent.backends import OpenAICompatBackend
        from slm_agent_lab.eval.report import load_runs, summarize
        from slm_agent_lab.eval.runner import run_eval
        backend = OpenAICompatBackend(args.base_url, args.model, tokenizer_id=args.tokenizer)
        run_dir = out_dir / f"harness_{args.tag}"
        run_eval(backend, tasks[: args.harness], run_dir, resume=False, extra_meta={"tag": args.tag})
        s = next(iter(summarize(load_runs([run_dir]))["models"].values()))
        print(json.dumps({"tag": args.tag, "accuracy": s["accuracy"], "malformed_rate": s["malformed_rate"],
                          "tool_error_rate": s["tool_error_rate"]}, indent=2))


if __name__ == "__main__":
    main()
