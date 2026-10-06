"""Model backends. Both render the *same* prompt with the model's own chat template and return raw text, so the
agent loop, parser and taxonomy are identical whether the model runs locally (Transformers) or behind vLLM.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

_CONTROL_TOKEN = re.compile(r"<\|(?!tool_call)[A-Za-z_]+\|>")


@dataclass
class GenOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    stop_reason: str  # "eos" | "length"


def _plain_messages(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Fallback rendering for tokenizers whose chat template has no tool support: tools are described in the
    system prompt and tool calls/results are carried as plain text."""
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "system" and i == 0 and tools:
            spec = "\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools)
            out.append({"role": "system", "content": m["content"] + "\n\nYou can call these tools:\n" + spec +
                        '\nTo call a tool, reply with <tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>.'})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            calls = "\n".join(
                "<tool_call>" + json.dumps({"name": c["function"]["name"], "arguments": c["function"]["arguments"]},
                                           ensure_ascii=False) + "</tool_call>" for c in m["tool_calls"])
            out.append({"role": "assistant", "content": (m.get("content") or "") + "\n" + calls})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": f"<tool_response>\n{m['content']}\n</tool_response>"})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class ChatRenderer:
    """Wraps a tokenizer's chat template; probes once whether it understands `tools=`."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.native_tools = self._probe()

    def _probe(self) -> bool:
        probe_tools = [{"type": "function", "function": {"name": "zzz_probe_tool", "description": "probe",
                                                           "parameters": {"type": "object", "properties": {}}}}]
        try:
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tools=probe_tools,
                                                      add_generation_prompt=True, tokenize=False)
            return "zzz_probe_tool" in text
        except Exception:
            return False

    def render(self, messages: list[dict], tools: list[dict] | None) -> str:
        if self.native_tools:
            try:
                return self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
            except Exception:
                pass
        return self.tokenizer.apply_chat_template(_plain_messages(messages, tools), add_generation_prompt=True, tokenize=False)

    def clean(self, text: str) -> str:
        for tok in (self.tokenizer.eos_token, self.tokenizer.pad_token, getattr(self.tokenizer, "bos_token", None)):
            if tok:
                text = text.replace(tok, "")
        return _CONTROL_TOKEN.sub("", text).strip()


class HFBackend:
    """Local Hugging Face Transformers backend (CPU or GPU), optionally with a merged LoRA adapter."""

    def __init__(self, model_id: str, adapter: str | None = None, dtype: str = "float32", device: str | None = None,
                 max_new_tokens: int = 256, num_threads: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id, self.adapter, self.max_new_tokens = model_id, adapter, max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if num_threads:
            torch.set_num_threads(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        except TypeError:  # transformers < 4.56 spelling
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.to(self.device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.renderer = ChatRenderer(self.tokenizer)
        self.label = model_id.split("/")[-1] + (f"+{adapter.rstrip('/\\').split('/')[-1].split(chr(92))[-1]}" if adapter else "")

    def render(self, messages: list[dict], tools: list[dict] | None) -> str:
        return self.renderer.render(messages, tools)

    def generate(self, prompt_text: str) -> GenOutput:
        import torch

        enc = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.device)
        n_prompt = int(enc["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        latency = time.perf_counter() - t0
        new_ids = out[0, n_prompt:]
        eos_ids = set(self.model.generation_config.eos_token_id if isinstance(self.model.generation_config.eos_token_id, list)
                      else [self.model.generation_config.eos_token_id])
        eos_ids.add(self.tokenizer.eos_token_id)
        ids = new_ids.tolist()
        n_gen = len(ids)
        for i, tid in enumerate(ids):
            if tid in eos_ids:
                n_gen = i + 1
                break
        text = self.tokenizer.decode(ids[:n_gen], skip_special_tokens=False)
        stop_reason = "eos" if any(t in eos_ids for t in ids[:n_gen]) else "length"
        return GenOutput(self.renderer.clean(text), n_prompt, n_gen, latency, stop_reason)


class OpenAICompatBackend:
    """Any OpenAI-compatible /v1/completions server (vLLM, TGI, llama.cpp). Prompt is rendered locally with the
    model's tokenizer so the dialect is identical to the HF backend."""

    def __init__(self, base_url: str, model: str, tokenizer_id: str | None = None, api_key: str = "EMPTY",
                 max_new_tokens: int = 256, timeout: float = 180.0):
        import requests
        from transformers import AutoTokenizer

        self._requests = requests
        self.base_url, self.model, self.max_new_tokens, self.timeout = base_url.rstrip("/"), model, max_new_tokens, timeout
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model)
        self.renderer = ChatRenderer(self.tokenizer)
        self.label = f"{model.split('/')[-1]}@{base_url}"
        self.model_id, self.adapter = model, None

    def render(self, messages: list[dict], tools: list[dict] | None) -> str:
        return self.renderer.render(messages, tools)

    def generate(self, prompt_text: str) -> GenOutput:
        payload = {"model": self.model, "prompt": prompt_text, "max_tokens": self.max_new_tokens, "temperature": 0,
                   "skip_special_tokens": False}
        t0 = time.perf_counter()
        r = self._requests.post(f"{self.base_url}/v1/completions", json=payload, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        latency = time.perf_counter() - t0
        choice = data["choices"][0]
        usage = data.get("usage", {})
        stop = "length" if choice.get("finish_reason") == "length" else "eos"
        return GenOutput(self.renderer.clean(choice["text"]), usage.get("prompt_tokens", 0),
                         usage.get("completion_tokens", 0), latency, stop)
