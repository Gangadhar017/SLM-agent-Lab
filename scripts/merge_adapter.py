"""Merge a LoRA adapter into its base model and save a standalone checkpoint (for vLLM / sharing).

  python scripts/merge_adapter.py --model ibm-granite/granite-4.0-350m --adapter results/sweep/granite-4.0-350m/r8_lr0.0002_s0 --out models/granite-350m-toolcall
"""

import argparse

import _bootstrap  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    try:
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print("saved merged model to", args.out)


if __name__ == "__main__":
    main()
