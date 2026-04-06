"""AWS Lambda handler for IaC repair using GGUF model.

Deploy with the 0.5B model (~300MB) for sub-$0.001/invocation cost.

Model loading:
  - Lambda layer: Place model at /opt/model.gguf
  - S3: Set MODEL_S3_BUCKET and MODEL_S3_KEY env vars
  - Bundled: Set MODEL_PATH env var

Example event:
    {
        "config": "resource aws_s3_bucket my_bucket {\\n  acl = \\"public\\"\\n}",
        "errors": ["Error: 'acl' argument is deprecated"],
        "format": "terraform"
    }

Response:
    {
        "repaired": "resource aws_s3_bucket my_bucket {\\n  ...\\n}",
        "original_errors": 1,
        "verified": true,
        "remaining_errors": 0
    }
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy-loaded repairer (persists across warm invocations)
_repairer = None


def _get_model_path() -> str:
    """Resolve the GGUF model path from environment or known locations."""
    # Explicit path
    if path := os.environ.get("MODEL_PATH"):
        return path

    # Lambda layer
    layer_path = "/opt/model.gguf"
    if Path(layer_path).exists():
        return layer_path

    # S3 download
    bucket = os.environ.get("MODEL_S3_BUCKET")
    key = os.environ.get("MODEL_S3_KEY")
    if bucket and key:
        local_path = "/tmp/model.gguf"
        if not Path(local_path).exists():
            import boto3

            logger.info("Downloading model from s3://%s/%s", bucket, key)
            s3 = boto3.client("s3")
            s3.download_file(bucket, key, local_path)
            logger.info("Model downloaded to %s", local_path)
        return local_path

    raise RuntimeError(
        "No model found. Set MODEL_PATH, place model at /opt/model.gguf, "
        "or set MODEL_S3_BUCKET + MODEL_S3_KEY."
    )


def _get_repairer():
    """Get or create the GGUFRepairer (cached across warm invocations)."""
    global _repairer
    if _repairer is None:
        from cloudgym.fixer.repairer import GGUFRepairer

        _repairer = GGUFRepairer(
            model_path=_get_model_path(),
            n_gpu_layers=0,
            max_tokens=4096,
        )
    return _repairer


def handler(event, context=None):
    """AWS Lambda entry point for IaC repair.

    Args:
        event: Dict with 'config' (str), 'errors' (list[str]), optional 'format' (str).
        context: Lambda context (unused).

    Returns:
        Dict with 'repaired' (str), 'verified' (bool), etc.
    """
    config = event.get("config", "")
    errors = event.get("errors", [])
    iac_format = event.get("format")

    if not config:
        return {"error": "Missing 'config' field", "statusCode": 400}

    if not errors:
        return {"repaired": config, "original_errors": 0, "verified": True, "remaining_errors": 0}

    repairer = _get_repairer()
    repaired = repairer.repair(config, errors)

    # Optional: verify the repair
    verified = False
    remaining_errors = -1
    try:
        from cloudgym.fixer.detector import IaCFormat, validate_content_sync

        fmt = IaCFormat(iac_format) if iac_format else None
        _, result = validate_content_sync(repaired, fmt)
        verified = result.valid
        remaining_errors = len(result.errors)
    except Exception:
        logger.exception("Verification failed")

    return {
        "repaired": repaired,
        "original_errors": len(errors),
        "verified": verified,
        "remaining_errors": remaining_errors,
    }
