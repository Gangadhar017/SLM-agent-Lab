# Stage 4 — serving and the accuracy/throughput trade-off

Goal: for each (model, adapter, precision) configuration, measure **both** serving performance (tokens/s, TTFT,
p50/p95 latency across concurrency) **and** agent task accuracy through the same harness. The interesting
question is not "is int4 faster" (it is) but *which failure category grows first when you quantize a small
tool-calling model* — does JSON formatting break before reasoning does?

> This stage needs a GPU. It was **not** executed on the development laptop (no NVIDIA GPU); the scripts and
> manifests are provided and tested only for syntax/import. Run on a cloud GPU node or a Kaggle T4 session.

## Precision sweep

| tag  | how to serve (vLLM)                                                       |
|------|----------------------------------------------------------------------------|
| fp16 | `--dtype float16`                                                          |
| fp8  | `--quantization fp8` (Hopper/Ada only)                                     |
| int8 | `--quantization bitsandbytes` or a pre-quantized W8A8 checkpoint           |
| int4 | AWQ/GPTQ checkpoint produced with `llm-compressor` or AutoAWQ, `--quantization awq` |

Merged LoRA adapters: `python scripts/merge_adapter.py` (or `PeftModel.merge_and_unload()` then `save_pretrained`)
and point vLLM at the merged directory; vLLM can also serve adapters directly with `--enable-lora`.

## Running the benchmark

```bash
# 1. serve (docker, single GPU)
docker run --gpus all -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai:latest \
  --model ibm-granite/granite-4.0-1b --dtype float16 --max-model-len 4096

# 2. throughput/latency across concurrency, plus the task harness through the server
python serve/bench.py --base-url http://localhost:8000 --model ibm-granite/granite-4.0-1b --tag fp16 \
  --concurrency 1,4,16,64 --harness 150
```

`results/serve/bench.csv` accumulates one row per (tag, concurrency); `results/serve/harness_<tag>/` holds the
trajectories, so `python scripts/make_report.py results/serve/harness_fp16 results/serve/harness_int4` gives the
failure taxonomy per precision.

## Kubernetes + Prometheus/Grafana

`k8s/vllm-deployment.yaml` deploys vLLM with a `ServiceMonitor`. Useful PromQL for the Grafana dashboard:

```promql
# generation throughput (tokens/s)
rate(vllm:generation_tokens_total[1m])
# time to first token, p95
histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[1m]))
# end-to-end request latency, p50
histogram_quantile(0.5, rate(vllm:e2e_request_latency_seconds_bucket[1m]))
# running / waiting requests
vllm:num_requests_running, vllm:num_requests_waiting
# KV cache utilisation
vllm:gpu_cache_usage_perc
```

## Stretch: one fused Triton kernel

A fused RMSNorm (or SwiGLU) kernel in OpenAI Triton benchmarked against PyTorch eager and `torch.compile` on the
same shapes as the served model. Not started; see the project README roadmap.
