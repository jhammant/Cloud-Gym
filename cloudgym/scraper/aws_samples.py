"""AWS CloudFormation sample template scraper."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from cloudgym.utils.config import GOLD_CF_DIR

logger = logging.getLogger(__name__)

# Official AWS CF sample repos on GitHub
AWS_CF_REPOS = [
    "aws-cloudformation/aws-cloudformation-templates",
    "awslabs/aws-cloudformation-templates",
    "aws-samples/aws-cloudformation-samples",
]


@dataclass
class AWSTemplateFile:
    """A scraped AWS CloudFormation sample template."""

    repo: str
    path: str
    content: str


@dataclass
class AWSSamplesScraper:
    """Scrapes AWS CloudFormation sample templates from official repos."""

    max_files: int = 200

    async def scrape(self) -> list[AWSTemplateFile]:
        """Fetch CF templates from AWS sample repositories."""
        results: list[AWSTemplateFile] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for repo in AWS_CF_REPOS:
                files = await self._scrape_repo(client, repo)
                results.extend(files)
                if len(results) >= self.max_files:
                    break
                await asyncio.sleep(1.0)

        return results[: self.max_files]

    async def _scrape_repo(
        self,
        client: httpx.AsyncClient,
        repo: str,
    ) -> list[AWSTemplateFile]:
        """Recursively scrape CF templates from a GitHub repo."""
        results: list[AWSTemplateFile] = []
        await self._walk_contents(client, repo, "", results)
        return results

    async def _walk_contents(
        self,
        client: httpx.AsyncClient,
        repo: str,
        path: str,
        results: list[AWSTemplateFile],
        depth: int = 0,
    ) -> None:
        """Walk repo contents recursively, collecting CF templates."""
        if depth > 3 or len(results) >= self.max_files:
            return

        api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
        try:
            resp = await client.get(api_url)
            resp.raise_for_status()
            items = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("Failed to list %s/%s: %s", repo, path, exc)
            return

        if not isinstance(items, list):
            return

        dirs = []
        download_tasks = []

        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            item_type = item.get("type", "")
            item_path = item.get("path", "")

            if item_type == "dir":
                dirs.append(item_path)
            elif item_type == "file" and self._is_cf_template(name):
                download_url = item.get("download_url", "")
                if download_url:
                    download_tasks.append(
                        self._download_template(client, repo, item_path, download_url)
                    )

        # Download files in parallel
        templates = await asyncio.gather(*download_tasks)
        results.extend(t for t in templates if t is not None)

        # Recurse into directories
        for d in dirs:
            await self._walk_contents(client, repo, d, results, depth + 1)
            await asyncio.sleep(0.5)  # Rate limiting

    def _is_cf_template(self, filename: str) -> bool:
        """Check if a filename looks like a CloudFormation template."""
        lower = filename.lower()
        cf_hints = ("template", "cfn", "cloudformation", "stack")
        is_yaml_json = lower.endswith((".yaml", ".yml", ".json"))
        has_hint = any(h in lower for h in cf_hints)
        return is_yaml_json and has_hint

    async def _download_template(
        self,
        client: httpx.AsyncClient,
        repo: str,
        path: str,
        url: str,
    ) -> AWSTemplateFile | None:
        """Download and verify a single CF template."""
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            content = resp.text
        except httpx.HTTPError:
            return None

        # Quick check: does it look like a CF template?
        if "AWSTemplateFormatVersion" not in content and "Resources" not in content:
            return None

        return AWSTemplateFile(repo=repo, path=path, content=content)


async def save_aws_samples(files: list[AWSTemplateFile]) -> int:
    """Save AWS sample templates to gold directory. Returns file count."""
    count = 0
    GOLD_CF_DIR.mkdir(parents=True, exist_ok=True)

    for f in files:
        safe_name = f"{f.repo}__{f.path}".replace("/", "_").replace("\\", "_")
        ext = ".yaml" if not f.path.endswith(".json") else ".json"
        if not safe_name.endswith(ext):
            safe_name += ext
        out_path = GOLD_CF_DIR / safe_name
        out_path.write_text(f.content, encoding="utf-8")
        count += 1

    return count
