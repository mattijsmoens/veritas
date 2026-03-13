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

from veritas.truth_guard import TruthGuard
from veritas.lora_export import LoRAExporter

__version__ = "0.1.1"


class Veritas:
    """
    End-to-end training data manager for the Truth Adapter pipeline.

    Wraps TruthGuard (runtime hallucination detection) and LoRAExporter
    (training data compilation) into a single interface. Provides methods
    to export training data, load adapters, and inspect data readiness.
    """

    def __init__(self, db_path="truth_guard.db", fact_ttl_days=7):
        """
        Args:
            db_path: Path to the TruthGuard SQLite database.
            fact_ttl_days: TTL in days for time-sensitive cached facts.
        """
        self.guard = TruthGuard(db_path=db_path, fact_ttl_days=fact_ttl_days)
        self.exporter = LoRAExporter(db_path=db_path)
        self._session_id = None

    def start_session(self, session_id):
        """Start a TruthGuard tracking session."""
        self._session_id = session_id
        self.guard.start_session(session_id)

    def end_session(self):
        """End the current TruthGuard tracking session."""
        if self._session_id:
            self.guard.end_session(self._session_id)
            self._session_id = None

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
        total = len(dataset)

        if total < min_pairs:
            return {
                "exported": False,
                "total_pairs": total,
                "min_required": min_pairs,
                "message": f"Not enough data yet. Have {total} pairs, need {min_pairs}.",
            }

        result = self.exporter.export_jsonl(output_path)
        return {
            "exported": True,
            "total_pairs": total,
            "path": result.get("output_path", output_path),
            "result": result,
        }

    def stats(self):
        """
        Return data readiness stats.

        Shows how many training pairs are available, broken down by type
        (negative corrections, positive verified, positive hedged, positive cited).
        """
        return self.exporter.stats

    def record_tool_use(self, tool_name, session_id=None):
        """Record that a verification tool was used in this session."""
        sid = session_id or self._session_id
        if not sid:
            raise ValueError("No active session. Call start_session() first or pass session_id.")
        self.guard.record_tool_use(sid, tool_name)

    def check_answer(self, answer, session_id=None):
        """
        Check an AI answer for unverified factual claims.

        Returns (allowed, reason) tuple. If the answer contains confident
        factual claims and no verification tool was used, it gets blocked.
        """
        sid = session_id or self._session_id
        if not sid:
            raise ValueError("No active session. Call start_session() first or pass session_id.")
        return self.guard.check_answer(sid, answer)

    def verify_fact(self, claim_text, source="manual", tool_used="MANUAL"):
        """Add a fact to the verified cache."""
        self.guard.store_verified_fact(claim_text, source, tool_used)


__all__ = ["Veritas", "TruthGuard", "LoRAExporter"]
