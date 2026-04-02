"""Terraform Registry module scraper."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from cloudgym.utils.config import GOLD_TF_DIR

logger = logging.getLogger(__name__)

REGISTRY_API = "https://registry.terraform.io/v1"


@dataclass
class RegistryModule:
    """A module scraped from the Terraform Registry."""

    namespace: str
    name: str
    provider: str
    version: str
    source_url: str
    configs: list[tuple[str, str]] = field(default_factory=list)  # (filename, content)


@dataclass
class RegistryScraper:
    """Scrapes verified modules from the Terraform Registry."""

    max_modules: int = 100
    providers: list[str] = field(default_factory=lambda: ["aws", "azurerm", "google"])

    async def scrape(self) -> list[RegistryModule]:
        """Fetch verified modules and download their source configs."""
        modules: list[RegistryModule] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for provider in self.providers:
                batch = await self._list_modules(client, provider)
                modules.extend(batch)
                if len(modules) >= self.max_modules:
                    break

            # Download source for each module
            tasks = [self._fetch_module_source(client, m) for m in modules]
            await asyncio.gather(*tasks)

        return [m for m in modules if m.configs]

    async def _list_modules(
        self, client: httpx.AsyncClient, provider: str
    ) -> list[RegistryModule]:
        """List verified modules for a given provider."""
        modules = []
        try:
            resp = await client.get(
                f"{REGISTRY_API}/modules",
                params={
                    "provider": provider,
                    "verified": "true",
                    "limit": 20,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Registry API failed for %s: %s", provider, exc)
            return modules

        data = resp.json()
        for mod in data.get("modules", []):
            modules.append(
                RegistryModule(
                    namespace=mod.get("namespace", ""),
                    name=mod.get("name", ""),
                    provider=mod.get("provider", ""),
                    version=mod.get("version", ""),
                    source_url=mod.get("source", ""),
                )
            )
        return modules

    async def _fetch_module_source(
        self, client: httpx.AsyncClient, module: RegistryModule
    ) -> None:
        """Fetch the download URL and retrieve .tf files from the source."""
        try:
            # Get the download URL from the registry
            resp = await client.get(
                f"{REGISTRY_API}/modules/"
                f"{module.namespace}/{module.name}/{module.provider}/"
                f"{module.version}/download",
                follow_redirects=True,
            )

            # The download endpoint returns a redirect header with the source archive
            download_url = resp.headers.get("X-Terraform-Get", "")
            if not download_url:
                return

            # If it's a GitHub source, try to get raw .tf files
            if "github.com" in download_url:
                await self._fetch_github_tf_files(client, download_url, module)

        except httpx.HTTPError as exc:
            logger.debug("Failed to fetch source for %s/%s: %s", module.namespace, module.name, exc)

    async def _fetch_github_tf_files(
        self,
        client: httpx.AsyncClient,
        github_url: str,
        module: RegistryModule,
    ) -> None:
        """Fetch .tf files from a GitHub repository URL."""
        # Convert GitHub URL to API URL for contents
        # e.g., https://github.com/org/repo -> api.github.com/repos/org/repo/contents
        parts = github_url.rstrip("/").split("github.com/")
        if len(parts) < 2:
            return

        repo_path = parts[1].split("?")[0].split("//")[0]
        api_url = f"https://api.github.com/repos/{repo_path}/contents"

        try:
            resp = await client.get(api_url)
            resp.raise_for_status()
            contents = resp.json()
        except (httpx.HTTPError, ValueError):
            return

        if not isinstance(contents, list):
            return

        for item in contents:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            if name.endswith(".tf") and item.get("type") == "file":
                download_url = item.get("download_url", "")
                if download_url:
                    try:
                        file_resp = await client.get(download_url)
                        file_resp.raise_for_status()
                        module.configs.append((name, file_resp.text))
                    except httpx.HTTPError:
                        continue


async def save_registry_modules(modules: list[RegistryModule]) -> int:
    """Save registry module configs to gold directory. Returns file count."""
    count = 0
    GOLD_TF_DIR.mkdir(parents=True, exist_ok=True)

    for module in modules:
        for filename, content in module.configs:
            safe_name = f"registry__{module.namespace}__{module.name}__{filename}"
            out_path = GOLD_TF_DIR / safe_name
            out_path.write_text(content, encoding="utf-8")
            count += 1

    return count
