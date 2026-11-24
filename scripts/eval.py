"""eval.py - Evaluate incident reports against golden set."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from incident_analyzer.evaluation import Evaluator, EvaluationMetrics
from incident_analyzer.generation import IncidentAgent
from incident_analyzer.ingestion.parser import parse_log_file, chunk_logs
from incident_analyzer.retrieval import build_retriever
from incident_analyzer.schemas import QueryRequest
from incident_analyzer.config import get_settings
from incident_analyzer.observability import setup_tracing


def main_eval(args: argparse.Namespace) -> int:
    """Evaluate incident reports against golden set."""
    print(f"Evaluating against golden set: {args.golden_set}")

    # Setup tracing
    setup_tracing()

    # Load settings
    settings = get_settings()

    # Parse and index logs once
    print("Parsing log files...")
    all_entries = []
    for log_file in Path(settings.log_dir).glob("*.log"):
        if log_file.name == "sample.log":
            continue
        print(f"  Parsing: {log_file.name}")
        entries = list(parse_log_file(log_file))
        all_entries.extend(entries)

    chunks = chunk_logs(all_entries)
    print(f"  Created {len(chunks)} chunks from {len(all_entries)} entries")

    # Build retriever
    retriever = build_retriever(chunks, use_semantic=settings.use_ollama)
    agent = IncidentAgent()

    # Initialize evaluator
    evaluator = Evaluator(golden_set_path=args.golden_set)

    # Evaluate each golden set entry
    print(f"\nRunning evaluation on {len(evaluator.golden_set)} entries...")
    queries_and_reports = []

    for entry in evaluator.golden_set:
        print(f"\n  Processing: {entry.id} - {entry.question[:60]}...")

        # Build query
        query = QueryRequest(
            question=entry.question,
            time_window_minutes=60,
            services=None,
            severity_min="error",
        )

        # Retrieve evidence
        evidence = retriever.search(query.question, top_k=5)
        print(f"    Retrieved {len(evidence)} chunks")

        # Generate report
        try:
            report = agent.analyze(query, evidence)
            print(f"    Report: {report.summary[:60]}... (conf: {report.confidence:.2f})")
            queries_and_reports.append((query, report))
        except Exception as e:
            print(f"    Error generating report: {e}")
            # Create a minimal failed report
            from incident_analyzer.schemas import IncidentReport, RetrievedChunk
            import uuid
            import datetime
            incident_id = f"INC-{datetime.datetime.now().strftime('%Y')}-{uuid.uuid4().hex[:8].upper()}"
            report = IncidentReport(
                incident_id=incident_id,
                summary=f"Error: {e}",
                root_cause="Error during analysis",
                confidence=0.0,
                affected_services=[],
                contributing_factors=[],
                timeline=[],
                evidence=[],
            )
            queries_and_reports.append((query, report))

    # Evaluate all
    print("\nEvaluating results...")
    start = time.time()
    metrics = evaluator.evaluate_all(queries_and_reports)
    metrics.eval_time_ms = int((time.time() - start) * 1000)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total entries: {metrics.total_entries}")
    print(f"Passed entries: {metrics.passed_entries}")
    print(f"Pass rate: {metrics.pass_rate:.1%}")
    print(f"Avg Groundedness: {metrics.overall_groundedness_avg:.2f}")
    print(f"Avg Faithfulness: {metrics.overall_faithfulness_avg:.2f}")
    print(f"Avg Answer Relevance: {metrics.overall_answer_relevance_avg:.2f}")
    print(f"Eval time: {metrics.eval_time_ms}ms")

    print("\nPer-entry results:")
    for result in metrics.per_entry:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.entry_id}: groundedness={result.groundedness_score:.2f}, faithfulness={result.faithfulness_score:.2f}, relevance={result.answer_relevance_score:.2f}")
        if result.missed_evidence_ids:
            print(f"      Missed evidence: {result.missed_evidence_ids}")
        if result.extra_evidence_ids:
            print(f"      Extra evidence: {result.extra_evidence_ids}")

    return 0 if metrics.pass_rate >= 0.7 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate against golden set")
    parser.add_argument("golden_set", help="Path to golden set JSONL file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main_eval(args))