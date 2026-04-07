# Cloud-Gym

Scalable Training Data Generation for Infrastructure-as-Code Repair via Environment Inversion.

Cloud-Gym generates (broken_config, error_message, fix) training pairs for IaC repair by applying **environment inversion** — taking working Terraform, CloudFormation, and OpenTofu configs and systematically breaking them using a defined fault taxonomy. It includes a benchmark (188 entries across 8 error categories) and fine-tuned models that run entirely on CPU.

## stackfix: AI-Powered IaC Repair

The `stackfix` CLI tool validates and repairs broken IaC files using fine-tuned local models. No API keys, no cloud costs, no data leaves your machine.

### Install

```bash
pip install cloud-gym[gguf]
```

### Download a Model

```bash
# Recommended: 3B Q4 (1.8 GB, 87% pass@1)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Tetsuto/iac-repair-3b-gguf', 'iac-repair-3b-q4.gguf', local_dir='.')
"
```

### Usage

```bash
# Check files for errors
stackfix check main.tf template.yaml

# Repair a broken file (show diff)
stackfix repair main.tf --backend gguf --model iac-repair-3b-q4.gguf

# Repair and apply fix in place
stackfix repair main.tf --apply --backend gguf --model iac-repair-3b-q4.gguf

# Explain errors in plain language
stackfix discuss main.tf --backend gguf --model iac-repair-3b-q4.gguf

# Pipe mode (stdin/stdout)
cat broken.tf | stackfix repair - --backend gguf --model iac-repair-3b-q4.gguf > fixed.tf

# Check all changed IaC files in git
stackfix git-diff --backend gguf --model iac-repair-3b-q4.gguf
```

### Models

| Model | Size | RAM | Speed (CPU) | pass@1 | HuggingFace |
|---|---|---|---|---|---|
| **7B Q4** | 4.5 GB | ~8 GB | ~20 tok/s | **0.926** | [Tetsuto/iac-repair-7b-gguf](https://huggingface.co/Tetsuto/iac-repair-7b-gguf) |
| **3B Q4** | 1.8 GB | ~4 GB | 49 tok/s | 0.867 | [Tetsuto/iac-repair-3b-gguf](https://huggingface.co/Tetsuto/iac-repair-3b-gguf) |
| **0.5B Q4** | 379 MB | ~800 MB | 127 tok/s | 0.723 | [Tetsuto/iac-repair-0.5b-gguf](https://huggingface.co/Tetsuto/iac-repair-0.5b-gguf) |

All models are fine-tuned Qwen2.5-Coder with LoRA, exported to GGUF. They run on any CPU (Linux, macOS, Windows).

### Backends

| Backend | Install | Platform | Use Case |
|---|---|---|---|
| `gguf` | `pip install cloud-gym[gguf]` | Any (CPU) | CI/CD, Lambda, servers |
| `mlx` | `pip install cloud-gym[mlx]` | Apple Silicon | Local dev on Mac |
| `ollama` | `pip install cloud-gym` + Ollama | Any | When Ollama is already running |

### CI/CD Integration

Add to your GitHub Actions workflow to catch IaC errors on every PR:

```yaml
- name: Check IaC
  run: |
    pip install cloud-gym[gguf]
    python -c "
    from huggingface_hub import hf_hub_download
    hf_hub_download('Tetsuto/iac-repair-3b-gguf', 'iac-repair-3b-q4.gguf', local_dir='.')
    "
    stackfix check **/*.tf **/*.yaml
```

See [examples/USE_CASES.md](examples/USE_CASES.md) for more deployment scenarios (pre-commit hooks, Lambda, pipeline integration).

### Pre-Commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: stackfix
        name: stackfix
        entry: stackfix pre-commit --backend gguf --model iac-repair-3b-q4.gguf
        language: python
        types_or: [terraform, yaml]
        additional_dependencies: ['cloud-gym[gguf]']
```

## Benchmark

188 entries across 8 error categories, 3 difficulty levels, and 2 formats (Terraform + CloudFormation).

### Results Summary

| Model | pass@1 | Terraform | CloudFormation | High | Medium | Low |
|---|---|---|---|---|---|---|
| **7B v2 fine-tuned** | **0.926** | 0.993 | 0.750 | 0.960 | 0.897 | 0.923 |
| 3B rank4 fine-tuned | 0.867 | 0.912 | 0.750 | 0.964 | 0.797 | 0.821 |
| qwen2.5-coder:7b (base) | 0.856 | 0.905 | 0.707 | 0.840 | 0.859 | 0.893 |
| 0.5B distilled | 0.723 | 0.775 | 0.590 | 0.809 | 0.648 | 0.731 |
| llama3.2:3b (base) | 0.641 | 0.734 | 0.361 | 0.684 | 0.636 | 0.533 |
| gemma-4-26b (base) | 0.009 | 0.000 | 0.032 | 0.000 | 0.004 | 0.051 |

Fine-tuning a 0.5B model outperforms a 26B base model by 80x.

## Training Data Generation

Cloud-Gym generates training data via environment inversion:

1. **Collect** working IaC configs from GitHub, Terraform Registry, AWS samples
2. **Break** them systematically using a fault taxonomy (28+ fault types across 8 categories)
3. **Validate** broken configs to capture real error messages
4. **Pair** (broken + errors) with the original working config as the gold fix

```bash
# Generate training data
cloud-gym taxonomy          # View fault types
python scripts/scrape.py    # Collect gold configs
cloud-gym invert            # Generate broken variants
cloud-gym export            # Export training pairs
```

## Project Structure

```text
cloudgym/
  taxonomy/     Fault type definitions (28+ types, 8 categories)
  scraper/      Gold config collection
  validator/    IaC validation wrappers (terraform, cfn-lint)
  inverter/     Fault injection engines
  generator/    Training data pipeline
  benchmark/    Evaluation harness
  fixer/        stackfix CLI tool + model backends
scripts/        Training, evaluation, and export scripts
examples/       Broken IaC examples + use case docs
```

## Supported Formats

- **Terraform** (`.tf`) — validated with `terraform validate`
- **CloudFormation** (`.yaml`, `.yml`, `.json`) — validated with `cfn-lint`
- **OpenTofu** (`.tf`) — same as Terraform

## License

MIT
