"""Ollama client wrapper for agentic fault injection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cloudgym.utils.config import InverterConfig

logger = logging.getLogger(__name__)


@dataclass
class OllamaClient:
    """Wrapper around the Ollama Python client for structured IaC prompting."""

    config: InverterConfig

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate a completion from the local Ollama instance."""
        try:
            import ollama as ollama_lib
        except ImportError:
            raise RuntimeError("ollama package not installed — run `pip install ollama`")

        client = ollama_lib.AsyncClient(host=self.config.ollama_base_url)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat(
            model=self.config.ollama_model,
            messages=messages,
        )
        return response["message"]["content"]

    async def inject_fault(
        self,
        config_content: str,
        fault_category: str,
        iac_format: str,
    ) -> str:
        """Ask the LLM to inject a realistic fault into an IaC config.

        Returns the modified (broken) config content.
        """
        system_prompt = (
            "You are an expert in Infrastructure-as-Code. Your task is to introduce "
            "a realistic, subtle bug into the given configuration. The bug should be "
            "the kind of mistake a real engineer might make. Return ONLY the modified "
            "configuration — no explanation, no markdown fences."
        )

        prompt = (
            f"Format: {iac_format}\n"
            f"Fault category: {fault_category}\n\n"
            f"Introduce a single {fault_category.lower()} fault into this config. "
            f"Make it realistic — something that would cause validation to fail "
            f"but isn't immediately obvious.\n\n"
            f"Original config:\n```\n{config_content}\n```\n\n"
            f"Return the modified config with the fault injected:"
        )

        return await self.generate(prompt, system=system_prompt)
