"""
Veritas — Truth Adapter Training Pipeline

Manages LoRA training data and adapters for teaching AI models
to prefer truthful, hedged responses over confident hallucinations.

Built on SovereignShield's TruthGuard and LoRAExporter pipeline.

Usage:
    from veritas import Veritas

    v = Veritas(db_path="truth_guard.db")
    v.export("training_data.jsonl")
    v.stats()
"""

from sovereign_shield import TruthGuard, LoRAExporter

__version__ = "0.1.0"


class Veritas:
    """
    End-to-end training data manager for the Truth Adapter pipeline.

    Wraps TruthGuard (runtime hallucination detection) and LoRAExporter
    (training data compilation) into a single interface. Provides methods
    to export training data, load adapters, and inspect data readiness.
    """

    def __init__(self, db_path="truth_guard.db", cache_ttl=3600):
        """
        Args:
            db_path: Path to the TruthGuard SQLite database.
            cache_ttl: Time-to-live in seconds for the verified fact cache.
        """
        self.guard = TruthGuard(db_path=db_path, cache_ttl=cache_ttl)
        self.exporter = LoRAExporter(db_path=db_path)

    def export(self, output_path="veritas_training.jsonl", min_pairs=10):
        """
        Export training data to JSONL format.

        Compiles blocked claims, verified facts, hedged responses, and cited
        answers from TruthGuard's database into training pairs suitable for
        LoRA fine-tuning with OpenAI, HuggingFace, or Unsloth.

        Args:
            output_path: Where to write the JSONL file.
            min_pairs: Minimum number of training pairs required before export.

        Returns:
            dict with export stats (total_pairs, path, breakdown by type).
        """
        dataset = self.exporter.compile_dataset()
        total = sum(len(v) for v in dataset.values())

        if total < min_pairs:
            return {
                "exported": False,
                "total_pairs": total,
                "min_required": min_pairs,
                "message": f"Not enough data yet. Have {total} pairs, need {min_pairs}.",
            }

        path = self.exporter.export_jsonl(output_path)
        return {
            "exported": True,
            "total_pairs": total,
            "path": path,
            "breakdown": {k: len(v) for k, v in dataset.items()},
        }

    def stats(self):
        """
        Return data readiness stats.

        Shows how many training pairs are available, broken down by type
        (negative corrections, positive verified, positive hedged, positive cited).
        """
        return self.exporter.stats()

    def record_tool_use(self, tool_name):
        """Record that a verification tool was used in this session."""
        self.guard.record_tool_use(tool_name)

    def check_answer(self, answer):
        """
        Check an AI answer for unverified factual claims.

        Returns (allowed, reason) tuple. If the answer contains confident
        factual claims and no verification tool was used, it gets blocked.
        """
        return self.guard.check_answer(answer)

    def verify_fact(self, claim):
        """Add a fact to the verified cache."""
        self.guard.add_verified_fact(claim)


__all__ = ["Veritas", "TruthGuard", "LoRAExporter"]
