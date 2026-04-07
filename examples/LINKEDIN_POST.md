# LinkedIn Post Draft

---

**I built an AI that fixes your broken Terraform and CloudFormation — and it runs on a CPU for free.**

We've all been there. A CloudFormation deploy fails at 2am because of a typo. Terraform plan catches a reference error after you've already context-switched. CI goes red because someone fat-fingered a resource name.

So I built **iac-fix** — an open-source CLI that validates IaC files, explains what's wrong in plain English, and generates verified fixes. The key difference: it runs a fine-tuned model locally on your laptop's CPU. No API keys. No cloud costs. No data leaving your machine.

**How it works, end to end:**

1. **Training data generation** — Cloud-Gym takes working Terraform/CloudFormation configs and systematically breaks them using a fault taxonomy (28 fault types across 8 categories: syntactic, semantic, reference, dependency, security, and more). This generates thousands of (broken config + error messages + correct fix) training pairs automatically.

2. **Fine-tuning** — We trained LoRA adapters on Qwen2.5-Coder models (0.5B, 3B, 7B) using these pairs. The key insight: getting the hyperparameters right matters enormously. Our first 7B attempt scored 3.5%. After fixing the LoRA layer coverage (8 → 20 out of 28 layers), sequence length (2048 → 4096), and scale (20 → 10), the same architecture hit **92.6%**.

3. **Deployment** — Models are exported to GGUF format and run via llama.cpp on any CPU. The 3B model (1.8 GB quantized) loads in 0.6 seconds and generates fixes at 49 tokens/sec. It fits in a GitHub Actions runner, a pre-commit hook, or an AWS Lambda function.

**The results speak for themselves:**

- Fine-tuned 7B: 92.6% pass@1 (99.3% on Terraform)
- Fine-tuned 3B: 86.7% pass@1 (1.8 GB, runs anywhere)
- Fine-tuned 0.5B: 72.3% pass@1 (379 MB, fits in Lambda)
- Gemma-4-26B (no fine-tuning): 0.9% pass@1

That's right — our 0.5B model (500M parameters) outperforms an unfine-tuned 26B model by **80x**. Fine-tuning on domain-specific data beats raw model size every time.

**What you can do with it today:**

```
# Validate
iac-fix check *.tf *.yaml

# Fix (shows diff, verifies the fix passes validation)
iac-fix repair main.tf --backend gguf --model iac-repair-3b-q4.gguf

# Explain errors in plain language
iac-fix discuss main.tf --backend gguf --model iac-repair-3b-q4.gguf

# Pipe mode for CI
cat broken.tf | iac-fix repair - --backend gguf --model model.gguf > fixed.tf
```

**Practical deployment options:**

- **Pre-commit hook**: Catches errors before they're committed. Fully offline.
- **GitHub Actions**: Free CI gate — block PRs with broken IaC configs.
- **AWS Lambda**: ~$0.0001 per invocation with the 0.5B model.
- **Local dev**: Just `pip install cloud-gym[gguf]` and go.

The whole pipeline is open source: data generation, training, evaluation benchmark, CLI tool, and model weights.

GitHub: github.com/jhammant/Cloud-Gym
Models: huggingface.co/Tetsuto/iac-repair-3b-gguf

What surprised me most: how little compute this took. The entire project — training data generation, fine-tuning across 5 model variants, compression experiments, GGUF export — ran on a single MacBook. Domain-specific fine-tuning makes small models punch way above their weight.

#InfrastructureAsCode #Terraform #CloudFormation #MachineLearning #DevOps #OpenSource #FineTuning #LLM

---

*Shorter version (if needed):*

**I fine-tuned a 0.5B model that fixes Terraform better than a 26B base model — and it runs on CPU for free.**

Cloud-Gym generates IaC training data via "environment inversion" (systematically breaking working configs), then fine-tunes small models to repair them.

Results: 7B → 92.6% pass@1. 3B → 86.7%. 0.5B → 72.3%. Gemma-4-26B without fine-tuning → 0.9%.

The 3B model is 1.8 GB, loads in 0.6s, and runs in GitHub Actions for free. No API keys, no cloud costs.

`pip install cloud-gym[gguf]` — open source, MIT licensed.

github.com/jhammant/Cloud-Gym

#Terraform #DevOps #MachineLearning #OpenSource
