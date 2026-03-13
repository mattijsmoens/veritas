"""Veritas CLI — command-line interface for training data management."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="veritas",
        description="Truth Adapter training data manager. Export TruthGuard data for LoRA fine-tuning.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export training data to JSONL")
    export_parser.add_argument(
        "-o", "--output",
        default="veritas_training.jsonl",
        help="Output JSONL file path (default: veritas_training.jsonl)",
    )
    export_parser.add_argument(
        "-d", "--db",
        default="truth_guard.db",
        help="Path to TruthGuard SQLite database (default: truth_guard.db)",
    )
    export_parser.add_argument(
        "--min-pairs",
        type=int,
        default=10,
        help="Minimum training pairs required before export (default: 10)",
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show training data readiness")
    stats_parser.add_argument(
        "-d", "--db",
        default="truth_guard.db",
        help="Path to TruthGuard SQLite database (default: truth_guard.db)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    from veritas import Veritas

    if args.command == "export":
        v = Veritas(db_path=args.db)
        result = v.export(output_path=args.output, min_pairs=args.min_pairs)
        if result["exported"]:
            print(f"Exported {result['total_pairs']} training pairs to {result['path']}")
            export_info = result.get("result", {})
            if "breakdown" in export_info:
                for pair_type, count in export_info["breakdown"].items():
                    print(f"  {pair_type}: {count}")
        else:
            print(result["message"])
            sys.exit(1)

    elif args.command == "stats":
        v = Veritas(db_path=args.db)
        stats = v.stats()
        print("Training Data Readiness")
        print("-" * 40)
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
