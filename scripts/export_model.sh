#!/usr/bin/env bash
# Fuse LoRA adapter into base model and register with Ollama
# Run after finetune.sh completes: ./scripts/export_model.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Step 1: Fuse LoRA adapter into base model ==="
.venv/bin/python3 -m mlx_lm fuse \
  --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
  --adapter-path data/models/iac-repair-adapter \
  --save-path data/models/iac-repair-fused

echo ""
echo "=== Step 2: Convert to GGUF ==="
# mlx_lm includes a GGUF converter
.venv/bin/python3 -m mlx_lm.gguf \
  --model data/models/iac-repair-fused \
  --output data/models/iac-repair.gguf \
  2>/dev/null || {
    echo "mlx_lm.gguf not available, trying llama.cpp convert..."
    echo "You may need to install llama-cpp-python or use llama.cpp's convert script."
    echo "Alternative: use the fused MLX model directly with mlx_lm.generate"
    exit 1
  }

echo ""
echo "=== Step 3: Create Ollama model ==="
cat > data/models/Modelfile <<'EOF'
FROM ./iac-repair.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"

SYSTEM """You are an Infrastructure-as-Code repair assistant. Fix the broken Terraform configuration below. Return ONLY the corrected HCL configuration with no explanation."""

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
EOF

cd data/models
ollama create cloudgym-repair -f Modelfile

echo ""
echo "=== Done! ==="
echo "Model registered as: cloudgym-repair"
echo "Test: ollama run cloudgym-repair 'Fix this: resource \"aws_vpc\" \"main\" { cidr_block = \"invalid\" }'"
