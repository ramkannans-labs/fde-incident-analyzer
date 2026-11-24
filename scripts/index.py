"""index.py - Index log files for incident analyzer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from incident_analyzer.evaluation import Evaluator
from incident_analyzer.ingestion.parser import parse_log_file, chunk_logs
from incident_analyzer.retrieval import build_retriever, HybridRetriever
from incident_analyzer.schemas import QueryRequest, IncidentReport, RetrievedChunk
from incident_analyzer.config import get_settings
from incident_analyzer.generation import IncidentAgent
from incident_analyzer.observability import setup_tracing


def main_index(args: argparse.Namespace) -> int:
    """Index log files for the incident analyzer."""
    print(f"Indexing logs from: {args.logs_dir}")

    # Setup tracing
    setup_tracing()

    # Initialize evaluator (golden set will be loaded when needed)
    evaluator = Evaluator()

    # Parse all log files
    all_entries = []
    for log_file in Path(args.logs_dir).glob("*.log"):
        print(f"Parsing: {log_file.name}")
        entries = list(parse_log_file(log_file))
        all_entries.extend(entries)

    # Chunk the logs
    chunks = chunk_logs(all_entries)

    # Build retriever
    settings = get_settings()
    retriever = build_retriever(chunks, use_semantic=settings.use_ollama)

    # Save chunks for later use (this is simplified - in real app this would persist)
    print(f"Indexed {len(chunks)} chunks from {len(all_entries)} log entries")

    # For now, just show we have indexed data
    print("Indexing complete. Ready for queries.")
    return 0


def main_ask(args: argparse.Namespace) -> int:
    """Ask a question about incident logs."""
    print(f"Asking: {args.question}")

    # Setup tracing
    setup_tracing()

    # Initialize components
    settings = get_settings()
    agent = IncidentAgent()

    # Parse query
    query = QueryRequest(
        question=args.question,
        time_window_minutes=args.time_window,
        services=args.services,
        severity_min=args.severity_min,
    )

    # For demo purposes, we'll simulate retrieval
    # In real implementation, this would use the indexed chunks
    simulated_chunks = []

    # Analyze the incident
    try:
        report: IncidentReport = agent.analyze(query, simulated_chunks)
        print("\n=== INCIDENT REPORT ===")
        print(f"Summary: {report.summary}")
        print(f"Root Cause: {report.root_cause}")
        print(f"Confidence: {report.confidence:.2f}")
        print(f"Services: {', '.join(report.affected_services)}")
        print(f"\nAffected Services: {report.affected_services}")
        print(f"Contributing Factors: {', '.join(report.contributing_factors) if report.contributing_factors else 'None'}")
        print(f"Fix Suggestion: {report.fix_suggestion or 'None'}")
        print(f"Requires Immediate Action: {report.requires_immediate_action}")
        print(f"\nEvidence:")
        for i, chunk in enumerate(report.evidence, 1):
            print(f"  [{i}] Score: {chunk.score:.2f} | Type: {chunk.match_type}")
            print(f"      Text: {chunk.chunk.text[:120]}...")
        return 0
    except Exception as e:
        print(f"Error analyzing incident: {e}")
        return 1


def main_eval(args: argparse.Namespace) -> int:
    """Evaluate incident reports against golden set."""
    print(f"Evaluating: {args.golden_set}")

    # Setup tracing
    setup_tracing()

    # Initialize evaluator
    evaluator = Evaluator(golden_set_path=args.golden_set)

    # For demo, we'd load actual golden set and compare against reports
    # This is a placeholder - real implementation would load actual data
    print("Golden set evaluation would run here")
    print(f"Loaded {len(evaluator.golden_set)} golden set entries")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incident Analyzer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    parser_index = subparsers.add_parser("index", help="Index log files")
    parser_index.add_argument("logs_dir", help="Directory containing log files to index")
    parser_index.set_defaults(func=main_index)

    # Ask command
    parser_ask = subparsers.add_parser("ask", help="Ask a question about incidents")
    parser_ask.add_argument("question", help="Natural language question")
    parser_ask.add_argument("--time-window", type=int, default=60, help="Time window in minutes")
    parser_ask.add_argument("--services", nargs="+", default=None, help="Filter to specific services")
    parser_ask.add_argument("--severity-min", default="error", help="Minimum severity level")
    parser_ask.set_defaults(func=main_ask)

    # Eval command
    parser_eval = subparsers.add_parser("eval", help="Evaluate against golden set")
    parser_eval.add_argument("golden_set", help="Path to golden set JSONL file")
    parser_eval.set_defaults(func=main_eval)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    commands = {"index": main_index, "ask": main_ask, "eval": main_eval}
    cmd = args.command
    if cmd and cmd in commands:
        sys.exit(commands[cmd](args))
    else:
        parser.print_help()
        sys.exit(1)