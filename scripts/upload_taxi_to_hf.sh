#!/usr/bin/env bash
# Upload the trained taxi-nl-3b GGUF + model card to HuggingFace.
# Run: ./scripts/upload_taxi_to_hf.sh   (assumes huggingface-cli login already done)
#
# Repository name follows the existing iac-repair pattern.

set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${HF_REPO:-Tetsuto/taxi-nl-3b-gguf}"
EXPORTS=data/models/exports

echo "=== uploading $REPO ==="
ls -lh $EXPORTS/taxi-nl-3b-q4.gguf $EXPORTS/taxi-nl-3b-f16.gguf $EXPORTS/MODEL_CARD.md

# Create the repo (idempotent)
.venv/bin/python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo(repo_id='$REPO', repo_type='model', exist_ok=True)
    print('repo ready')
except Exception as e:
    print(f'create_repo: {e}')
"

# Upload each file
.venv/bin/python3 -c "
from huggingface_hub import HfApi
api = HfApi()
files = [
    ('$EXPORTS/taxi-nl-3b-q4.gguf',  'taxi-nl-3b-q4.gguf'),
    ('$EXPORTS/taxi-nl-3b-f16.gguf', 'taxi-nl-3b-f16.gguf'),
    ('$EXPORTS/MODEL_CARD.md',       'README.md'),
]
for local, remote in files:
    print(f'uploading {local} -> {remote} ...')
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id='$REPO', repo_type='model')
print('done')
"

echo "=== https://huggingface.co/$REPO ==="
