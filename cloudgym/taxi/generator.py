"""NL→Taxi generator backed by MLX or GGUF model with the SFT v2 adapter.

Same backend abstraction as cloudgym/fixer/repairer.py — just different
system prompt and a post-process for the in-context schema dedup.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# System prompt — must match what the model was trained with in
# scripts/format_taxi_finetuning.py to keep distribution alignment.
TAXI_SYSTEM_PROMPT = (
    "You translate natural-language requirements into idiomatic Taxi schema code. "
    "Taxi is the schema language used by Orbital (orbitalhq.com). "
    "Return ONLY the Taxi source. Do not include prose, explanation, or markdown fences."
)

DEFAULT_BASE_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
DEFAULT_ADAPTER_PATH = "data/models/taxi-nl-adapter-v2"
DEFAULT_GGUF_REPO = "Tetsuto/taxi-nl-3b-gguf"
DEFAULT_GGUF_FILE = "taxi-nl-3b-q4.gguf"


# top-level declaration regex — matches `type/model/enum/annotation/service NAME`
_DECL_RE = re.compile(
    r"^\s*(?:closed\s+)?(type|model|enum|annotation|service)\s+([A-Za-z_]\w*)\b",
    re.MULTILINE,
)


def _strip_fences(text: str) -> str:
    """Remove markdown fences and chat tokens from model output."""
    for stop in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
        if stop in text:
            text = text[: text.index(stop)]
    text = text.strip()
    m = re.match(r"^```(?:taxi|kotlin)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_redeclared(output: str, in_context_schema: str) -> str:
    """Drop blocks from `output` whose declared symbol is already declared in
    `in_context_schema`. The fine-tuned model dutifully replays in-context
    schemas — strict validators reject the duplicate. This dedup is part of
    the production inference path.
    """
    if not in_context_schema:
        return output
    ctx_names = {m.group(2) for m in _DECL_RE.finditer(in_context_schema)}
    if not ctx_names:
        return output
    lines = output.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _DECL_RE.match(lines[i])
        if m and m.group(2) in ctx_names:
            if "{" not in lines[i]:
                i += 1
                continue
            depth = lines[i].count("{") - lines[i].count("}")
            j = i + 1
            while j < len(lines) and depth > 0:
                depth += lines[j].count("{") - lines[j].count("}")
                j += 1
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).strip()


def _build_user_prompt(description: str, schema: Optional[str]) -> str:
    if not schema:
        return description.strip()
    return (
        "Existing schema in scope:\n```taxi\n"
        + schema.strip()
        + "\n```\n\nTask: "
        + description.strip()
        + "\n\nReturn only the Taxi code that satisfies the task. Do not repeat the existing schema."
    )


# --- backends ---------------------------------------------------------------

class MLXGenerator:
    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
        temp: float = 0.0,
        max_tokens: int = 1500,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.temp = temp
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load
        adapter = self.adapter_path if (self.adapter_path and Path(self.adapter_path).exists()) else None
        logger.info("Loading MLX model %s (adapter: %s)", self.base_model, adapter)
        self._model, self._tokenizer = load(self.base_model, adapter_path=adapter)

    def generate(self, description: str, schema: Optional[str] = None) -> str:
        self._ensure_loaded()
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        messages = [
            {"role": "system", "content": TAXI_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(description, schema)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        sampler = make_sampler(temp=self.temp)
        text = generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=self.max_tokens, sampler=sampler,
        )
        return _strip_redeclared(_strip_fences(text), schema or "")


class GGUFGenerator:
    def __init__(
        self,
        gguf_path: str | Path,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        temp: float = 0.0,
        max_tokens: int = 1500,
    ) -> None:
        self.gguf_path = str(gguf_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temp = temp
        self.max_tokens = max_tokens
        self._llm = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama
        kw = dict(model_path=self.gguf_path, n_ctx=self.n_ctx,
                  n_gpu_layers=self.n_gpu_layers, verbose=False)
        if self.n_threads is not None:
            kw["n_threads"] = self.n_threads
        logger.info("Loading GGUF model %s", self.gguf_path)
        self._llm = Llama(**kw)

    def generate(self, description: str, schema: Optional[str] = None) -> str:
        self._ensure_loaded()
        messages = [
            {"role": "system", "content": TAXI_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(description, schema)},
        ]
        resp = self._llm.create_chat_completion(
            messages=messages, temperature=self.temp, max_tokens=self.max_tokens,
        )
        text = resp["choices"][0]["message"]["content"]
        return _strip_redeclared(_strip_fences(text or ""), schema or "")


def resolve_gguf_model(model: str | None) -> str:
    """Resolve a GGUF path, auto-downloading from HuggingFace if needed."""
    if model and Path(model).exists():
        return model
    from huggingface_hub import hf_hub_download
    repo = DEFAULT_GGUF_REPO
    filename = DEFAULT_GGUF_FILE
    return hf_hub_download(repo_id=repo, filename=filename)
