"""Python wrapper around the Kotlin validator shim.

The shim is a long-running JVM process exposing POST /validate on a local port.
We launch it on demand, hold the subprocess for the lifetime of the validator
instance, and submit Taxi sources via HTTP for structured compile errors.

Why a JVM shim instead of Orbital's API:
  Orbital's workspace API silently absorbs unresolved type references (it
  treats them as lazily-resolvable cross-package symbols). The taxilang
  Compiler.validate() — which the shim wraps — runs the full strict type
  checker. Confirmed in P1 smoke tests: dangling refs surface as
  errorCode="UnresolvedType".

Reference:
  Compiler entry point — data/upstream/taxilang/compiler/src/main/java/lang/taxi/Compiler.kt:415–426
  Shim source         — cloudgym/taxi/_validator_jvm/src/main/kotlin/cloudgym/taxi/Shim.kt
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
from urllib import request as _urlreq
from urllib.error import URLError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JAR = REPO_ROOT / "cloudgym/taxi/_validator_jvm/target/taxi-validator-shim.jar"
DEFAULT_JAVA_HOME = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"


@dataclass
class CompilationError:
    line: int
    char: int
    severity: str
    detailMessage: str
    errorCode: Optional[str] = None
    sourceName: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return self.severity.lower() == "error"


@dataclass
class ValidationResult:
    is_valid: bool
    error_count: int
    warning_count: int
    errors: list[CompilationError] = field(default_factory=list)

    def messages(self, severity: str | None = None) -> list[str]:
        rows = self.errors
        if severity is not None:
            rows = [e for e in rows if e.severity.lower() == severity.lower()]
        return [e.detailMessage for e in rows]


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TaxiValidator:
    """Long-running JVM-backed Taxi validator. Safe to share across threads (the
    shim has an internal 8-thread executor)."""

    def __init__(
        self,
        *,
        jar_path: Path | str | None = None,
        port: int | None = None,
        java_home: str | None = None,
        startup_timeout: float = 15.0,
    ) -> None:
        self.jar_path = Path(jar_path) if jar_path else DEFAULT_JAR
        if not self.jar_path.exists():
            raise FileNotFoundError(
                f"Validator JAR not found at {self.jar_path}. Build it via:\n"
                f"  cd cloudgym/taxi/_validator_jvm && mvn clean package"
            )
        self.port = port or _free_port()
        self.java_home = java_home or os.environ.get("JAVA_HOME") or DEFAULT_JAVA_HOME
        self._proc: subprocess.Popen[bytes] | None = None
        self._url = f"http://127.0.0.1:{self.port}"
        self._startup_timeout = startup_timeout

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            return
        env = os.environ.copy()
        env["JAVA_HOME"] = self.java_home
        env["PATH"] = f"{self.java_home}/bin:{env.get('PATH', '')}"
        env["PORT"] = str(self.port)
        java = f"{self.java_home}/bin/java"
        if not Path(java).exists():
            java = "java"  # fall back to whatever's on PATH
        self._proc = subprocess.Popen(
            [java, "-jar", str(self.jar_path)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self.stop)
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            try:
                with _urlreq.urlopen(f"{self._url}/health", timeout=1) as r:
                    if r.status == 200:
                        return
            except (URLError, ConnectionError, OSError):
                pass
            time.sleep(0.2)
        self.stop()
        raise RuntimeError(f"Validator shim failed to come up within {self._startup_timeout}s")

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
        finally:
            self._proc = None

    def __enter__(self) -> "TaxiValidator":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- API -------------------------------------------------------------

    def validate(self, source: str, source_name: str = "input.taxi") -> ValidationResult:
        if self._proc is None:
            self.start()
        req = _urlreq.Request(
            f"{self._url}/validate",
            data=source.encode("utf-8"),
            headers={"Content-Type": "text/plain", "X-Source-Name": source_name},
            method="POST",
        )
        with _urlreq.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read())
        return ValidationResult(
            is_valid=payload["isValid"],
            error_count=payload["errorCount"],
            warning_count=payload["warningCount"],
            errors=[CompilationError(**e) for e in payload["errors"]],
        )

    def validate_multi(self, sources: list[tuple[str, str]]) -> ValidationResult:
        """Compile multiple .taxi sources together as one package. Each entry is
        (source_name, content). Use this for sibling files that reference each
        other across files, or for benchmark prompts that include an in-context
        schema fragment + a target snippet."""
        if self._proc is None:
            self.start()
        body = json.dumps({
            "sources": [{"name": n, "content": c} for n, c in sources],
        }).encode("utf-8")
        req = _urlreq.Request(
            f"{self._url}/validate-multi",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urlreq.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read())
        return ValidationResult(
            is_valid=payload["isValid"],
            error_count=payload["errorCount"],
            warning_count=payload["warningCount"],
            errors=[CompilationError(**e) for e in payload["errors"]],
        )

    def validate_many(
        self, sources: Iterable[tuple[str, str]]
    ) -> list[ValidationResult]:
        """Sequential validate; the shim has internal threading so concurrency
        gains come from running multiple callers, not multiple sources per call."""
        return [self.validate(src, name) for name, src in sources]


# Module-level convenience singleton -----------------------------------------

_default: TaxiValidator | None = None


def validate(source: str, source_name: str = "input.taxi") -> ValidationResult:
    """Convenience wrapper using a process-wide singleton validator."""
    global _default
    if _default is None:
        _default = TaxiValidator()
        _default.start()
    return _default.validate(source, source_name)


__all__ = ["TaxiValidator", "ValidationResult", "CompilationError", "validate"]
