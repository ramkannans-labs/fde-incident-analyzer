"""Evaluation harness for incident-analyzer — golden set runner + metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field
from ..schemas import (
    IncidentReport,
)


class GoldenSetEntry(BaseModel):
    """A single golden set entry: question + expected answer."""

    id: str = Field(description="Unique entry identifier")
    question: str = Field(description="Natural language question about logs")
    expected_incident_id: str = Field(description="Expected incident ID in answer")
    expected_root_cause: str = Field(description="Expected root cause string")
    expected_confidence: float = Field(
        ge=0.0, le=1.0, description="Expected confidence score"
    )
    expected_affected_services: List[str] = Field(
        default_factory=list, description="Expected affected services"
    )
    golden_evidence_ids: List[str] = Field(
        default_factory=list, description="Chunk IDs that should be referenced"
    )
    # Evaluation flags
    groundedness_pass: Optional[bool] = Field(default=None)
    faithfulness_pass: Optional[bool] = Field(default=None)


class EvaluationResult(BaseModel):
    """Result of evaluating one golden set entry against agent output."""

    entry_id: str = Field(description="Golden set entry ID")
    question: str = Field(description="The question")
    report: IncidentReport = Field(description="Agent's output report")
    groundedness_score: float = Field(ge=0.0, le=1.0, description="How well citations are supported")
    faithfulness_score: float = Field(ge=0.0, le=1.0, description="How well answer reflects evidence")
    answer_relevance_score: float = Field(ge=0.0, le=1.0, description="How relevant answer is to question")
    matched_evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence chunk IDs that matched"
    )
    missed_evidence_ids: List[str] = Field(
        default_factory=list, description="Golden evidence IDs not in agent output"
    )
    extra_evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence IDs in agent output not in golden"
    )
    passed: bool = Field(description="Whether overall metrics pass thresholds")
    evaluated_at: datetime = Field(default_factory=datetime.now)
    notes: str = Field(default="", description="Human note on evaluation")


class EvaluationMetrics(BaseModel):
    """Aggregated metrics across golden set."""

    total_entries: int = Field(description="Total golden set entries")
    passed_entries: int = Field(description="Entries where passed=True")
    overall_groundedness_avg: float = Field(ge=0.0, le=1.0, description="Average groundedness")
    overall_faithfulness_avg: float = Field(ge=0.0, le=1.0, description="Average faithfulness")
    overall_answer_relevance_avg: float = Field(
        ge=0.0, le=1.0, description="Average answer relevance"
    )
    groundedness_at_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    faithfulness_at_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    pass_rate: float = Field(description="Pass rate (passed/total)")
    eval_time_ms: int = Field(description="Total evaluation time in ms")
    per_entry: List[EvaluationResult] = Field(
        default_factory=list, description="Per-entry results"
    )


class Evaluator:
    """Evaluates agent output against golden set."""

    def __init__(self, golden_set_path: str = "tests/golden_set.jsonl"):
        self.golden_set_path = Path(golden_set_path)
        self._golden_set: List[GoldenSetEntry] = []
        self._load_golden_set()

    def _load_golden_set(self) -> None:
        """Load golden set from JSONL file."""
        if not self.golden_set_path.exists():
            # No golden set yet - that's OK, evaluator starts empty
            return

        with open(self.golden_set_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entry = GoldenSetEntry(**data)
                    self._golden_set.append(entry)

    @property
    def golden_set(self) -> List[GoldenSetEntry]:
        """Return the loaded golden set."""
        return self._golden_set

    def evaluate(
        self,
        entry: GoldenSetEntry,
        report: IncidentReport,
    ) -> EvaluationResult:
        """Evaluate one entry against agent output."""
        # Simple heuristics for evaluation metrics
        groundedness = self._compute_groundedness(entry, report)
        faithfulness = self._compute_faithfulness(entry, report)
        answer_relevance = self._compute_answer_relevance(entry, report)

        matched = self._compute_matched_evidence(entry, report)
        missed = self._compute_missed_evidence(entry, report)
        extra = self._compute_extra_evidence(entry, report)

        passed = (
            groundedness >= 0.7
            and faithfulness >= 0.7
            and answer_relevance >= 0.7
        )

        return EvaluationResult(
            entry_id=entry.id,
            question=entry.question,
            report=report,
            groundedness_score=groundedness,
            faithfulness_score=faithfulness,
            answer_relevance_score=answer_relevance,
            matched_evidence_ids=matched,
            missed_evidence_ids=missed,
            extra_evidence_ids=extra,
            passed=passed,
        )

    def evaluate_all(
        self,
        queries_and_reports: List[tuple[QueryRequest, IncidentReport]],
    ) -> EvaluationMetrics:
        """Evaluate all entries and return aggregated metrics."""
        results = []
        for query, report in queries_and_reports:
            # Find matching golden set entry
            entry = self._find_entry(query.question)
            if entry is None:
                # No golden set entry - create a basic result
                results.append(
                    EvaluationResult(
                        entry_id=f"auto_{len(results)}",
                        question=query.question,
                        report=report,
                        groundedness_score=0.5,
                        faithfulness_score=0.5,
                        answer_relevance_score=0.5,
                        passed=False,
                    )
                )
                continue

            result = self.evaluate(entry, report)
            results.append(result)

        return self._aggregate(results)

    def _find_entry(self, question: str) -> Optional[GoldenSetEntry]:
        """Find matching golden set entry by question similarity."""
        if not self._golden_set:
            return None

        # Try exact match first
        for entry in self._golden_set:
            if entry.question.lower().strip() == question.lower().strip():
                return entry

        # Try substring match
        for entry in self._golden_set:
            if entry.question.lower().strip() in question.lower().strip() or \
               question.lower().strip() in entry.question.lower().strip():
                return entry

        return None

    def _compute_groundedness(self, entry: GoldenSetEntry, report: IncidentReport) -> float:
        """Compute groundedness: how well report claims are supported by evidence."""
        if not report.evidence:
            return 0.0

        # Check that each claim in summary/root_cause/fix_suggestion
        # can be traced to at least one evidence chunk
        claims = [
            report.summary,
            report.root_cause,
        ]
        if report.fix_suggestion:
            claims.append(report.fix_suggestion)

        supported_claims = 0
        for claim in claims:
            if claim and claim.strip():
                # Check if any evidence chunk text contains relevant keywords
                is_supported = False
                for ev in report.evidence:
                    ev_text = ev.chunk.text.lower()
                    claim_lower = claim.lower()
                    # Simple keyword overlap check
                    claim_words = set(claim_lower.split())
                    ev_words = set(ev_text.split())
                    if claim_words & ev_words:
                        is_supported = True
                        break
                if is_supported:
                    supported_claims += 1

        if not claims:
            return 0.5  # Neutral - no claims to check
        return supported_claims / len(claims)

    def _compute_faithfulness(self, entry: GoldenSetEntry, report: IncidentReport) -> float:
        """Compute faithfulness: how well the answer reflects only what's in evidence."""
        # Check that report claims don't contradict evidence
        if not report.evidence:
            return 0.0

        faithfulness_score = 1.0

        # Check that root_cause is present in evidence
        if report.root_cause and report.root_cause.strip():
            root_cause_lower = report.root_cause.lower()
            all_ev_text = " ".join(
                ev.chunk.text.lower() for ev in report.evidence
            )
            if root_cause_lower not in all_ev_text:
                # Root cause not directly in evidence - deduct
                faithfulness_score -= 0.3

        # Check that summary doesn't add info not in evidence
        if report.summary and report.summary.strip():
            summary_lower = report.summary.lower()
            all_ev_text = " ".join(
                ev.chunk.text.lower() for ev in report.evidence
            )
            # If summary has words not in evidence at all
            ev_words = set(all_ev_text.split())
            summary_words = set(summary_lower.split())
            extra_words = summary_words - ev_words - {
                "the", "a", "an", "and", "or", "but", "in", "on", "to",
                "for", "of", "with", "as", "by", "at", "from"
            }
            if extra_words:
                faithfulness_score -= 0.2 * min(len(extra_words) / 20, 1.0)

        return max(faithfulness_score, 0.0)

    def _compute_answer_relevance(
        self, entry: GoldenSetEntry, report: IncidentReport
    ) -> float:
        """Compute answer relevance: how well the answer addresses the question."""
        if not report.summary:
            return 0.0

        question_lower = entry.question.lower()
        summary_lower = report.summary.lower()

        # Check key question terms appear in summary
        question_words = set(question_lower.split())
        summary_words = set(summary_lower.split())

        # Remove stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "to",
            "for", "of", "with", "as", "by", "at", "from", "what",
            "how", "why", "is", "are", "was", "were", "wasn't",
        }
        question_terms = question_words - stop_words
        summary_terms = summary_words - stop_words

        if not question_terms:
            return 0.5  # Neutral if no question terms

        overlap = len(question_terms & summary_terms) / len(question_terms)
        return min(overlap + 0.2, 1.0)  # Base + bonus for good coverage

    def _compute_matched_evidence(
        self, entry: GoldenSetEntry, report: IncidentReport
    ) -> List[str]:
        """Find evidence chunk IDs from report that are in golden set."""
        matched = []
        golden_ids = set(entry.golden_evidence_ids)
        report_ids = {ev.chunk.chunk_id for ev in report.evidence}

        for chunk_id in report_ids:
            if chunk_id in golden_ids:
                matched.append(chunk_id)

        return matched

    def _compute_missed_evidence(
        self, entry: GoldenSetEntry, report: IncidentReport
    ) -> List[str]:
        """Find golden evidence IDs not in report."""
        golden_ids = set(entry.golden_evidence_ids)
        report_ids = {ev.chunk.chunk_id for ev in report.evidence}

        missed = golden_ids - report_ids
        return list(missed)

    def _compute_extra_evidence(
        self, entry: GoldenSetEntry, report: IncidentReport
    ) -> List[str]:
        """Find evidence IDs in report not in golden set."""
        report_ids = {ev.chunk.chunk_id for ev in report.evidence}
        golden_ids = set(entry.golden_evidence_ids)

        extra = report_ids - golden_ids
        return list(extra)

    def _aggregate(
        self, results: List[EvaluationResult]
    ) -> EvaluationMetrics:
        """Aggregate per-entry results into metrics."""
        total = len(results)
        if total == 0:
            return EvaluationMetrics(
                total_entries=0,
                passed_entries=0,
                overall_groundedness_avg=0.0,
                overall_faithfulness_avg=0.0,
                overall_answer_relevance_avg=0.0,
                eval_time_ms=0,
                per_entry=results,
            )

        passed = sum(1 for r in results if r.passed)
        groundedness_avg = sum(r.groundedness_score for r in results) / total
        faithfulness_avg = sum(r.faithfulness_score for r in results) / total
        answer_relevance_avg = sum(r.answer_relevance_score for r in results) / total

        return EvaluationMetrics(
            total_entries=total,
            passed_entries=passed,
            overall_groundedness_avg=groundedness_avg,
            overall_faithfulness_avg=faithfulness_avg,
            overall_answer_relevance_avg=answer_relevance_avg,
            groundedness_at_threshold=None,
            faithfulness_at_threshold=None,
            pass_rate=passed / total if total > 0 else 0.0,
            eval_time_ms=0,
            per_entry=results,
        )