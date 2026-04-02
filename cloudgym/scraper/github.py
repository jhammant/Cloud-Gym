"""GitHub scraper for Terraform and CloudFormation configurations."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from cloudgym.utils.config import GOLD_CF_DIR, GOLD_TF_DIR, ScraperConfig

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
GITHUB_SEARCH_CODE = f"{GITHUB_API}/search/code"
GITHUB_SEARCH_REPOS = f"{GITHUB_API}/search/repositories"

# Patterns that suggest secrets — skip these files
SECRET_PATTERNS = re.compile(
    r"(AKIA[0-9A-Z]{16}|aws_secret_access_key|password\s*=\s*\".{8,}\")",
    re.IGNORECASE,
)


@dataclass
class ScrapedFile:
    """A single scraped IaC configuration file."""

    repo_full_name: str
    file_path: str
    content: str
    format: str  # "terraform" or "cloudformation"
    sha: str = ""


@dataclass
class GitHubScraper:
    """Scrapes GitHub for Terraform and CloudFormation configs."""

    config: ScraperConfig = field(default_factory=ScraperConfig)
    _seen_hashes: set[str] = field(default_factory=set, repr=False)

    @property
    def _headers(self) -> dict[str, str]:
        token = self.config.github_token or os.environ.get("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def scrape_terraform(self) -> list[ScrapedFile]:
        """Search GitHub for Terraform .tf files."""
        results: list[ScrapedFile] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in self.config.tf_search_queries:
                files = await self._search_code(
                    client,
                    query=f"{query} extension:tf",
                    fmt="terraform",
                )
                results.extend(files)
                if len(results) >= self.config.max_repos:
                    break
                # Respect rate limits
                await asyncio.sleep(2.0)
        return results

    async def scrape_cloudformation(self) -> list[ScrapedFile]:
        """Search GitHub for CloudFormation templates."""
        results: list[ScrapedFile] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in self.config.cf_search_queries:
                files = await self._search_code(
                    client,
                    query=f"{query} extension:yaml",
                    fmt="cloudformation",
                )
                results.extend(files)

                # Also search JSON CF templates
                files = await self._search_code(
                    client,
                    query=f"{query} extension:json",
                    fmt="cloudformation",
                )
                results.extend(files)

                if len(results) >= self.config.max_repos:
                    break
                await asyncio.sleep(2.0)
        return results

    async def scrape_all(self) -> list[ScrapedFile]:
        """Run all GitHub scraping tasks."""
        tf_files, cf_files = await asyncio.gather(
            self.scrape_terraform(),
            self.scrape_cloudformation(),
        )
        return tf_files + cf_files

    async def _search_code(
        self,
        client: httpx.AsyncClient,
        query: str,
        fmt: str,
        per_page: int = 30,
    ) -> list[ScrapedFile]:
        """Search GitHub code API and download matching files."""
        results: list[ScrapedFile] = []

        try:
            resp = await client.get(
                GITHUB_SEARCH_CODE,
                params={"q": query, "per_page": per_page},
                headers=self._headers,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                logger.warning("GitHub API rate limit hit, pausing")
                await asyncio.sleep(60)
                return results
            logger.error("GitHub search failed: %s", exc)
            return results

        data = resp.json()
        items = data.get("items", [])
        logger.info("GitHub search '%s' returned %d items", query, len(items))

        for item in items:
            sha = item.get("sha", "")
            if sha in self._seen_hashes:
                continue

            raw_url = item.get("html_url", "").replace(
                "github.com", "raw.githubusercontent.com"
            ).replace("/blob/", "/")

            if not raw_url:
                continue

            content = await self._download_raw(client, raw_url)
            if content is None:
                continue

            if not self._passes_filters(content, fmt):
                continue

            self._seen_hashes.add(sha)
            results.append(
                ScrapedFile(
                    repo_full_name=item.get("repository", {}).get("full_name", ""),
                    file_path=item.get("path", ""),
                    content=content,
                    format=fmt,
                    sha=sha,
                )
            )

        return results

    async def _download_raw(self, client: httpx.AsyncClient, url: str) -> str | None:
        """Download raw file content from GitHub."""
        try:
            resp = await client.get(url, headers=self._headers, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text
            if len(text) > self.config.max_file_size_kb * 1024:
                return None
            return text
        except httpx.HTTPError:
            return None

    def _passes_filters(self, content: str, fmt: str) -> bool:
        """Check if file content passes quality filters."""
        # Skip files with obvious secrets
        if SECRET_PATTERNS.search(content):
            logger.debug("Skipping file with potential secrets")
            return False

        # Skip very small files
        if len(content.strip()) < 50:
            return False

        if fmt == "terraform":
            return self._filter_terraform(content)
        elif fmt == "cloudformation":
            return self._filter_cloudformation(content)
        return False

    def _filter_terraform(self, content: str) -> bool:
        """Check Terraform-specific quality criteria."""
        resource_count = content.count("resource ")
        data_count = content.count("data ")
        module_count = content.count("module ")
        total = resource_count + data_count + module_count
        return total >= self.config.min_resources

    def _filter_cloudformation(self, content: str) -> bool:
        """Check CloudFormation-specific quality criteria."""
        has_version = "AWSTemplateFormatVersion" in content
        has_resources = "Resources:" in content or '"Resources"' in content
        if not (has_version or has_resources):
            return False

        # Count resources roughly
        resource_lines = content.count("Type: AWS::") + content.count('"Type": "AWS::')
        return resource_lines >= self.config.min_resources


async def save_scraped_files(files: list[ScrapedFile]) -> dict[str, int]:
    """Save scraped files to the gold directories. Returns count per format."""
    counts = {"terraform": 0, "cloudformation": 0}

    for f in files:
        if f.format == "terraform":
            out_dir = GOLD_TF_DIR
            ext = ".tf"
        else:
            out_dir = GOLD_CF_DIR
            ext = ".yaml" if not f.file_path.endswith(".json") else ".json"

        out_dir.mkdir(parents=True, exist_ok=True)

        # Use repo name + file path as unique filename
        safe_name = f"{f.repo_full_name}__{f.file_path}".replace("/", "_").replace("\\", "_")
        if not safe_name.endswith(ext):
            safe_name += ext

        out_path = out_dir / safe_name
        out_path.write_text(f.content, encoding="utf-8")
        counts[f.format] += 1

    return counts
