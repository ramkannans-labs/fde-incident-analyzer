"""ask.py - Ask a question about incident logs."""

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

    # Parse and index logs
    print("Parsing log files...")
    all_entries = []
    for log_file in Path(settings.log_dir).glob("*.log"):
        print(f"  Parsing: {log_file.name}")
        entries = list(parse_log_file(log_file))
        all_entries.extend(entries)

    chunks = chunk_logs(all_entries)
    print(f"  Created {len(chunks)} chunks from {len(all_entries)} entries")

    # Build retriever and search
    retriever = build_retriever(chunks, use_semantic=settings.use_ollama)
    evidence = retriever.search(query.question, top_k=5)
    print(f"  Retrieved {len(evidence)} relevant chunks")

    # Analyze the incident
    try:
        report: IncidentReport = agent.analyze(query, evidence)
        print("\n=== INCIDENT REPORT ===")
        print(f"Summary: {report.summary}")
        print(f"Root Cause: {report.root_cause}")
        print(f"Confidence: {report.confidence:.2f}")
        print(f"Services: {', '.join(report.affected_services) if report.affected_services else 'None'}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask about incident logs")
    parser.add_argument("question", help="Natural language question about logs")
    parser.add_argument("--time-window", type=int, default=60, help="Time window in minutes")
    parser.add_argument("--services", nargs="+", default=None, help="Filter to specific services")
    parser.add_argument("--severity-min", default="error", help="Minimum severity level")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main_ask(args))