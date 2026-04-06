# iac-fix Use Cases

AI-powered Infrastructure-as-Code repair using a fine-tuned local model.
Runs on CPU anywhere — no GPU required, no API keys, no cloud costs.

## Quick Start

```bash
# Install
pip install cloud-gym[gguf]

# Download the model (1.8 GB, one-time)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Tetsuto/iac-repair-3b-gguf', 'iac-repair-3b-q4.gguf', local_dir='.')
"

# Fix a broken Terraform file
iac-fix repair main.tf --backend gguf --model iac-repair-3b-q4.gguf
```

---

## Use Case 1: CI/CD Pipeline Gate

**Scenario:** Validate IaC files on every pull request. Block merges with broken configs, auto-suggest fixes in PR comments.

```yaml
# .github/workflows/iac-fix.yml
name: iac-fix
on:
  pull_request:
    paths: ['**/*.tf', '**/*.yaml', '**/*.yml']

jobs:
  iac-fix:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }

      - name: Install
        run: pip install cloud-gym[gguf]

      - name: Download model
        run: |
          python -c "
          from huggingface_hub import hf_hub_download
          hf_hub_download('Tetsuto/iac-repair-3b-gguf', 'iac-repair-3b-q4.gguf', local_dir='.')
          "

      - name: Check IaC files
        run: |
          git diff --name-only origin/${{ github.base_ref }}...HEAD \
            | grep -E '\.(tf|yaml|yml)$' \
            | xargs -r iac-fix check
```

**Cost:** Free (GitHub Actions free tier). Model downloads once per run (~30s).

---

## Use Case 2: Pre-Commit Hook

**Scenario:** Catch and fix IaC errors before they're even committed. The model runs locally — no network needed after initial model download.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: iac-fix
        name: iac-fix
        entry: iac-fix pre-commit --backend gguf --model ./iac-repair-3b-q4.gguf
        language: python
        types_or: [terraform, yaml]
        additional_dependencies: ['cloud-gym[gguf]']
```

**How it works:**
1. Developer commits `main.tf` with a typo
2. Pre-commit runs `iac-fix`, detects the error
3. Model generates the fix and writes it in place
4. Developer re-stages and commits the corrected file

---

## Use Case 3: Incident Response — Explain What Went Wrong

**Scenario:** A CloudFormation deployment failed. Use `iac-fix discuss` to get a plain-language explanation of what's wrong, why it matters, and how to fix it.

```bash
# Get a detailed explanation of errors
iac-fix discuss failed-template.yaml --backend gguf --model iac-repair-3b-q4.gguf
```

**Output:**
```text
failed-template.yaml (cloudformation)
1 error(s), 0 warning(s)

╭─────────────────────── Analysis ───────────────────────╮
│ 1. **What's wrong:** The template references            │
│    'InterntGateway' but the resource is named           │
│    'InternetGateway' — a typo.                          │
│                                                         │
│ 2. **Why it matters:** CloudFormation will fail to      │
│    resolve the !Ref and the stack creation/update       │
│    will fail, leaving your VPC without internet.        │
│                                                         │
│ 3. **How to fix:** Change `!Ref InterntGateway` to     │
│    `!Ref InternetGateway`                               │
│                                                         │
│ 4. **Best practice:** Use IDE autocomplete for          │
│    resource references, or add cfn-lint to CI.          │
╰────────────────────────────────────────────────────────╯
```

---

## Use Case 4: Pipeline Integration (stdin/stdout)

**Scenario:** Integrate into any build pipeline using Unix pipes. Feed broken config in, get fixed config out.

```bash
# Fix inline — pipe broken config through iac-fix
cat broken.tf | iac-fix repair - --backend gguf --model iac-repair-3b-q4.gguf > fixed.tf

# Use in a script
terraform validate main.tf 2>&1 || \
  iac-fix repair main.tf --apply --backend gguf --model iac-repair-3b-q4.gguf

# Chain with other tools
iac-fix repair main.tf --backend gguf --model iac-repair-3b-q4.gguf -o /dev/stdout \
  | terraform fmt -
```

---

## Use Case 5: Git Diff Check

**Scenario:** Before pushing, check all changed IaC files in your working tree. Auto-fix and re-stage.

```bash
# Check changed files
iac-fix git-diff --backend gguf --model iac-repair-3b-q4.gguf

# Auto-fix and stage
iac-fix git-diff --apply --backend gguf --model iac-repair-3b-q4.gguf
```

---

## Use Case 6: AWS Lambda (Serverless API)

**Scenario:** Expose IaC repair as an API endpoint for other tools/services to call. Costs ~$0.0001 per invocation.

```python
# Deploy with: the 0.5B model fits in Lambda's 10GB limit
# Set env: MODEL_S3_BUCKET=my-bucket MODEL_S3_KEY=iac-repair-0.5b-q4.gguf

# Call the Lambda
import boto3, json
lambda_client = boto3.client('lambda')
response = lambda_client.invoke(
    FunctionName='iac-fix',
    Payload=json.dumps({
        "config": open("broken.tf").read(),
        "errors": ["Reference to undeclared resource"],
        "format": "terraform",
    }),
)
result = json.loads(response['Payload'].read())
print(result['repaired'])    # Fixed config
print(result['verified'])     # True if fix passes validation
```

---

## Model Options

| Model | Size | pass@1 | Best For |
|---|---|---|---|
| `iac-repair-3b-q4.gguf` | 1.8 GB | 0.867 | CI/CD, pre-commit, servers |
| `iac-repair-3b-f16.gguf` | 5.8 GB | 0.867 | Max precision (same accuracy) |
| `iac-repair-0.5b-q4.gguf` | 379 MB | 0.723 | Lambda, edge, resource-constrained |
| `iac-repair-0.5b-f16.gguf` | 948 MB | 0.723 | 0.5B full precision |

All models available at:
- [Tetsuto/iac-repair-3b-gguf](https://huggingface.co/Tetsuto/iac-repair-3b-gguf)
- [Tetsuto/iac-repair-0.5b-gguf](https://huggingface.co/Tetsuto/iac-repair-0.5b-gguf)

## Supported Formats

- **Terraform** (`.tf`) — validated with `terraform validate`
- **CloudFormation** (`.yaml`, `.yml`, `.json`) — validated with `cfn-lint`
- **OpenTofu** (`.tf`) — same as Terraform

## How It Works

The model is a fine-tuned Qwen2.5-Coder with a LoRA adapter trained on 7,500+
IaC error/fix pairs across 8 error categories (syntactic, semantic, reference,
dependency, security, intrinsic, cross-resource, provider). It runs entirely
locally — no API calls, no data leaves your machine.
