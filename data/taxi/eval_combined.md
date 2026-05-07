# Taxi Benchmark — combined report

All models scored against `data/taxi/benchmark.jsonl` (100 entries:
40 easy, 30 schema_aware, 30 open_ended across 11 domains).

Metrics:
- **pass** = compile pass rate (strict taxilang validator)
- **decl** = top-level declaration recall vs gold
- **field** = field name+type recall vs gold
- **easy / sa / oe** = pass rate per difficulty bucket

| Model | n | pass | decl | field | s/q | easy / sa / oe |
|---|--:|--:|--:|--:|--:|---|
| lmstudio-api:qwen/qwen3-coder-next [plain] | 100 | 19% | 46% | 50% | 2.1 | 18 / 40 / 0 |
| ollama:qwen2.5-coder:32b [plain] | 100 | 30% | 35% | 53% | 5.9 | 28 / 63 / 0 |
| ollama:qwen2.5-coder:7b [plain] | 100 | 24% | 51% | 48% | 1.5 | 38 / 30 / 0 |
| lmstudio-api:qwen/qwen3-coder-next [stuffed] | 100 | 72% | 82% | 57% | 2.6 | 90 / 90 / 30 |
| ollama:qwen2.5-coder:32b [stuffed] | 100 | 80% | 82% | 55% | 6.7 | 90 / 97 / 50 |
| ollama:qwen2.5-coder:7b [stuffed] | 100 | 67% | 77% | 54% | 1.8 | 70 / 87 / 43 |

## Context-stuffing delta (pass rate)

| Model | plain | stuffed | Δ |
|---|--:|--:|--:|
| lmstudio-api:qwen/qwen3-coder-next | 19% | 72% | +53pp |
| ollama:qwen2.5-coder:32b | 30% | 80% | +50pp |
| ollama:qwen2.5-coder:7b | 24% | 67% | +43pp |
