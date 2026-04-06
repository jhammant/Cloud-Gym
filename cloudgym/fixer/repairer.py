"""Model-based IaC repair engine."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPAIR_SYSTEM_PROMPT = (
    "You are an expert Infrastructure-as-Code engineer. "
    "Fix the broken configuration below. Return ONLY the fixed configuration "
    "with no explanation, no markdown fences, and no comments about the fix."
)

DISCUSS_SYSTEM_PROMPT = (
    "You are an expert Infrastructure-as-Code engineer and teacher. "
    "Analyze the configuration and validation errors below. Explain:\n"
    "1. What each error means in plain language\n"
    "2. Why it's a problem (what would happen in production)\n"
    "3. How to fix it\n"
    "4. Best practices to avoid this in future\n"
    "Be concise but thorough."
)

# Default model/adapter paths (relative to package or absolute)
DEFAULT_BASE_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
DEFAULT_ADAPTER_PATH = "data/models/iac-repair-adapter"


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences and chat tokens from model output."""
    for stop in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
        if stop in text:
            text = text[:text.index(stop)]

    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _build_prompt(broken_config: str, errors: list[str]) -> str:
    """Build the repair prompt from config content and error messages."""
    error_text = "\n".join(errors) if errors else "Unknown error"
    return (
        f"This IaC configuration has validation errors:\n\n"
        f"Errors:\n{error_text}\n\n"
        f"Broken config:\n```\n{broken_config}\n```\n\n"
        f"Return the fixed configuration:"
    )


def _build_discuss_prompt(config: str, errors: list[str]) -> str:
    """Build the discussion prompt for explaining errors."""
    error_text = "\n".join(errors) if errors else "No validation errors"
    return (
        f"Analyze this IaC configuration:\n\n"
        f"Validation errors:\n{error_text}\n\n"
        f"Configuration:\n```\n{config}\n```\n\n"
        f"Explain what's wrong and how to fix it:"
    )


class MLXRepairer:
    """Repair IaC configs using a local MLX fine-tuned model."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
        temp: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.temp = temp
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return

        from mlx_lm import load

        adapter = self.adapter_path if Path(self.adapter_path).exists() else None
        logger.info("Loading model %s (adapter: %s)", self.base_model, adapter)
        self._model, self._tokenizer = load(
            self.base_model, adapter_path=adapter
        )
        logger.info("Model loaded.")

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response with the given prompts."""
        self._ensure_loaded()

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        sampler = make_sampler(temp=self.temp)
        return generate(
            self._model, self._tokenizer, prompt=prompt_text,
            max_tokens=self.max_tokens, sampler=sampler,
        )

    def repair(self, broken_config: str, errors: list[str]) -> str:
        """Generate a repaired config from broken config + errors."""
        return _strip_markdown_fences(
            self._generate(REPAIR_SYSTEM_PROMPT, _build_prompt(broken_config, errors))
        )

    def discuss(self, config: str, errors: list[str]) -> str:
        """Explain errors and suggest fixes in natural language."""
        return self._generate(DISCUSS_SYSTEM_PROMPT, _build_discuss_prompt(config, errors))


class GGUFRepairer:
    """Repair IaC configs using a GGUF model via llama-cpp-python (cross-platform CPU)."""

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        temp: float = 0.3,
        max_tokens: int = 4096,
        n_ctx: int = 8192,
    ):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.temp = temp
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self._llm = None

    def _ensure_loaded(self):
        """Lazy-load the GGUF model."""
        if self._llm is not None:
            return

        from llama_cpp import Llama

        logger.info("Loading GGUF model from %s", self.model_path)
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        logger.info("GGUF model loaded.")

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response via llama.cpp chat completion."""
        self._ensure_loaded()

        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temp,
            max_tokens=self.max_tokens,
        )
        return response["choices"][0]["message"]["content"]

    def repair(self, broken_config: str, errors: list[str]) -> str:
        """Generate a repaired config from broken config + errors."""
        return _strip_markdown_fences(
            self._chat(REPAIR_SYSTEM_PROMPT, _build_prompt(broken_config, errors))
        )

    def discuss(self, config: str, errors: list[str]) -> str:
        """Explain errors and suggest fixes in natural language."""
        return self._chat(DISCUSS_SYSTEM_PROMPT, _build_discuss_prompt(config, errors))


class OllamaRepairer:
    """Repair IaC configs using an Ollama-served model."""

    def __init__(self, model: str = "qwen2.5-coder:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat request to Ollama."""
        import httpx

        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def repair(self, broken_config: str, errors: list[str]) -> str:
        """Generate a repaired config via Ollama API."""
        return _strip_markdown_fences(
            self._chat(REPAIR_SYSTEM_PROMPT, _build_prompt(broken_config, errors))
        )

    def discuss(self, config: str, errors: list[str]) -> str:
        """Explain errors and suggest fixes in natural language."""
        return self._chat(DISCUSS_SYSTEM_PROMPT, _build_discuss_prompt(config, errors))
