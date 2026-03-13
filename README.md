# Veritas — Truth Adapter Training Pipeline

Training data manager and adapter loader for teaching AI models to prefer truthful, hedged responses over confident hallucinations. Built on SovereignShield's TruthGuard pipeline.

## What This Does

AI models confidently state things that aren't true. SovereignShield's TruthGuard catches these hallucinations at runtime by blocking answers that contain confident factual claims without tool-backed verification. But blocking bad answers is reactive — the long-term goal is to make the model stop hallucinating in the first place.

Veritas is the training side of that pipeline. It takes everything TruthGuard has collected — blocked claims, verified facts, hedged responses, cited answers — and compiles them into JSONL training pairs. You feed those pairs into a LoRA fine-tuning tool (OpenAI API, HuggingFace, Unsloth), and the model learns to prefer truthful, hedged responses over confident guesses. Over time, the model internalizes the behavior and stops needing TruthGuard to catch it.

## Install

```bash
pip install veritas-truth-adapter
```

## Quick Start

### Python API

```python
from veritas import Veritas

v = Veritas(db_path="truth_guard.db")

# Check data readiness
print(v.stats())

# Export training data when ready
result = v.export("training_data.jsonl")
if result["exported"]:
    print(f"Exported {result['total_pairs']} training pairs")

# Use TruthGuard directly through Veritas
v.record_tool_use("SEARCH")
allowed, reason = v.check_answer("The capital of France is Paris.")
```

### CLI

```bash
# Check how much training data is available
veritas stats -d truth_guard.db

# Export training data to JSONL
veritas export -d truth_guard.db -o training_data.jsonl
```

## Training Pair Types

Veritas generates four types of training pairs:

**Negative corrections** — The model made a confident claim that TruthGuard blocked. The training pair shows the model what it should have said instead (a hedged version of the same claim). This teaches the model to stop making unverified assertions.

**Positive verified** — The model made a factual claim AND used a verification tool first. The training pair reinforces this behavior — the model gets positive signal for checking its facts before stating them.

**Positive hedged** — The model expressed appropriate uncertainty ("I believe", "as far as I know") instead of stating something as fact. The training pair rewards this behavior.

**Positive cited** — The model included a source or reference for its claim. The training pair rewards citing evidence.

## How the Pipeline Works

```
TruthGuard (runtime)          Veritas (training)              Fine-tuning (external)
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────────────┐
│ AI generates     │     │ Compile blocked      │     │ Feed JSONL into:         │
│ answer           │     │ claims + verified    │     │ - OpenAI fine-tuning API │
│       ↓          │     │ facts + hedged       │     │ - HuggingFace/Unsloth   │
│ Check for        │     │ responses + cited    │     │ - Any JSONL-compatible   │
│ confidence       │────→│ answers into JSONL   │────→│   LoRA trainer          │
│ markers          │     │ training pairs       │     │       ↓                  │
│       ↓          │     │                      │     │ Model learns to prefer   │
│ Block or Allow   │     │ veritas export       │     │ truthful responses       │
│       ↓          │     │                      │     │                          │
│ Log to SQLite    │     │                      │     │                          │
└──────────────────┘     └─────────────────────┘     └──────────────────────────┘
```

## Configuration

```python
v = Veritas(
    db_path="truth_guard.db",   # Path to TruthGuard's SQLite database
    cache_ttl=3600,             # Verified fact cache TTL in seconds (default: 1 hour)
)

# Export with minimum pair threshold
result = v.export(
    output_path="training_data.jsonl",
    min_pairs=10,               # Won't export unless you have at least this many pairs
)
```

## Dependencies

- `sovereign-shield>=1.2.0` — Provides TruthGuard and LoRAExporter

## License

BSL 1.1 — See [LICENSE](LICENSE)
