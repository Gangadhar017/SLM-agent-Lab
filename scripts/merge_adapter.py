"""Merge a LoRA adapter into its base model and write a standalone Hugging Face checkpoint (for GGUF conversion,
vLLM, or sharing).

Default mode ("raw") merges directly in the base checkpoint's safetensors: W' = W + (alpha / r) * B @ A for every
adapted module, keeping the Hub's tensor layout and all of the original support files (config, tokenizer,
chat template). This avoids transformers' save path, which (as of transformers 5.x) rewrites this architecture's
weights/config in a form that llama.cpp's converter does not read back correctly. `--mode peft` uses
PeftModel.merge_and_unload() + save_pretrained() instead.

  python scripts/merge_adapter.py --model ibm-granite/granite-4.0-350m --adapter results/sweep/granite-4.0-350m/r8_lr0.0002_s0 --out /tmp/granite-350m-r8
"""

import argparse
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.py"]


def resolve_base_dir(model: str) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, allow_patterns=WEIGHT_PATTERNS))


def load_base_tensors(base_dir: Path) -> dict:
    from safetensors.torch import load_file
    index = base_dir / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        tensors = {}
        for s in shards:
            tensors.update(load_file(str(base_dir / s)))
        return tensors
    return load_file(str(base_dir / "model.safetensors"))


def merge_raw(model: str, adapter: str, out: str) -> dict:
    import torch
    from safetensors.torch import load_file, save_file

    base_dir, adapter_dir, out_dir = resolve_base_dir(model), Path(adapter), Path(out)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    base = load_base_tensors(base_dir)
    ad = load_file(str(adapter_dir / "adapter_model.safetensors"))

    n_merged, max_delta = 0, 0.0
    for k, A in ad.items():
        if not k.endswith("lora_A.weight"):
            continue
        prefix = k[: -len(".lora_A.weight")]
        B = ad[prefix + ".lora_B.weight"]
        target = prefix.replace("base_model.model.", "", 1) + ".weight"
        if target not in base:
            raise KeyError(f"adapter module {prefix} has no base weight {target}")
        W = base[target]
        delta = scaling * (B.float() @ A.float())
        if delta.shape != W.shape:
            raise ValueError(f"shape mismatch for {target}: {delta.shape} vs {W.shape}")
        base[target] = (W.float() + delta).to(W.dtype)
        max_delta = max(max_delta, float(delta.abs().max()))
        n_merged += 1
    if n_merged == 0:
        raise ValueError("no lora_A/lora_B pairs found in the adapter")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(base, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    copied = []
    for f in base_dir.iterdir():
        if f.is_file() and f.name != "model.safetensors" and not f.name.startswith("model-") and f.suffix != ".index":
            if f.name == "model.safetensors.index.json":
                continue
            shutil.copy2(f, out_dir / f.name)
            copied.append(f.name)
    return {"merged_modules": n_merged, "scaling": scaling, "max_abs_delta": max_delta, "copied": copied,
            "out": str(out_dir)}


def merge_peft(model: str, adapter: str, out: str, dtype: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    try:
        base = AutoModelForCausalLM.from_pretrained(model, dtype=torch_dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model).save_pretrained(out)
    return {"out": out, "mode": "peft"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model id or local directory")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["raw", "peft"], default="raw")
    ap.add_argument("--dtype", default="float16", help="(peft mode only)")
    args = ap.parse_args()
    info = merge_raw(args.model, args.adapter, args.out) if args.mode == "raw" else merge_peft(args.model, args.adapter, args.out, args.dtype)
    print(json.dumps(info, indent=2))
    print("saved merged model to", args.out)


if __name__ == "__main__":
    main()
