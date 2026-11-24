"""PydanticAI agent for incident analysis — structured, cited, observable."""

from __future__ import annotations

import os
import uuid
import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from ..schemas import IncidentReport, RetrievedChunk, QueryRequest
from ..config import get_settings


def _get_model():
    """Get the model to use — falls back to test model if no API key."""
    settings = get_settings()

    # Check if we have an Anthropic API key
    if settings.llm_model.startswith("anthropic:") and not os.getenv("ANTHROPIC_API_KEY"):
        # Use test model for local development without API key
        return "test"
    return settings.llm_model


class IncidentAgent:
    """PydanticAI agent that analyzes incidents from log chunks."""

    def __init__(self):
        model = _get_model()
        self._use_mock = model == "test"

        if not self._use_mock:
            self.agent: Agent[None, IncidentReport] = Agent(
                model,
                output_type=IncidentReport,
                system_prompt=(
                    "You are an expert SRE incident analyst. "
                    "Your job is to analyze log data and produce a structured incident report. "
                    "You MUST ground every claim in the provided log evidence. "
                    "Never speculate beyond what the logs show. "
                    "If the logs do not contain enough information to determine root cause, "
                    "set confidence to a low value and explain what is missing. "
                    "Always include the exact log lines that support your conclusions in the evidence field."
                ),
            )
        else:
            self.agent = None

    def analyze(self, query: QueryRequest, evidence: List[RetrievedChunk]) -> IncidentReport:
        """Analyze an incident given a query and retrieved evidence."""
        if self._use_mock:
            return self._mock_analyze(query, evidence)

        prompt = self._build_prompt(query, evidence)
        result = self.agent.run_sync(prompt)
        return result.output

    def _mock_analyze(self, query: QueryRequest, evidence: List[RetrievedChunk]) -> IncidentReport:
        """Mock analysis for testing without LLM - produces grounded output from evidence."""
        incident_id = f"INC-{datetime.datetime.now().strftime('%Y')}-{uuid.uuid4().hex[:8].upper()}"

        if not evidence:
            return IncidentReport(
                incident_id=incident_id,
                summary="No log evidence available to analyze",
                root_cause="Insufficient data - no relevant log chunks retrieved",
                confidence=0.1,
                affected_services=[],
                contributing_factors=["No evidence retrieved"],
                timeline=[],
                evidence=[],
                fix_suggestion="Check log ingestion and retrieval pipeline",
                requires_immediate_action=False,
            )

        # Extract key information from evidence
        all_text = " ".join(e.chunk.text.lower() for e in evidence)
        all_services = set()
        all_levels = set()
        for e in evidence:
            all_services.update(e.chunk.services)
            all_levels.add(e.chunk.severity)

        # Generate incident ID
        incident_id = f"INC-{datetime.datetime.now().strftime('%Y')}-{uuid.uuid4().hex[:8].upper()}"

        # Build timeline from evidence
        timeline = []
        for e in evidence:
            if e.chunk.time_range:
                timeline.append({
                    "time": e.chunk.time_range[0],
                    "event": f"Log entries from {', '.join(e.chunk.services)} ({e.chunk.severity})",
                    "source": e.chunk.chunk_id,
                })

        # Determine root cause based on evidence content
        root_cause_parts = []
        if "payment failed" in all_text or "card declined" in all_text:
            root_cause_parts.append("Payment declined by card issuer")
        if "authentication failed" in all_text or "invalid credentials" in all_text:
            root_cause_parts.append("Authentication failures due to invalid credentials")
        if "database connection timeout" in all_text:
            root_cause_parts.append("Database connection timeout")
        if "disk" in all_text and "failed" in all_text:
            root_cause_parts.append("Disk read failure")

        root_cause = "; ".join(root_cause_parts) if root_cause_parts else "Multiple issues detected in logs"

        # Determine affected services
        affected_services = list(all_services) if all_services else ["unknown"]

        # Contributing factors
        factors = []
        if "warning" in all_levels:
            factors.append("Warning-level events preceding errors")
        if "timeout" in all_text:
            factors.append("Timeout conditions detected")
        if "connection" in all_text and ("fail" in all_text or "timeout" in all_text):
            factors.append("Connection failures")

        # Confidence based on evidence quality
        has_errors = "error" in all_levels or "critical" in all_levels
        confidence = 0.7 if has_errors and len(evidence) >= 2 else 0.4 if evidence else 0.1

        # Fix suggestion
        fix_suggestions = []
        if "payment failed" in all_text:
            fix_suggestions.append("Investigate payment gateway integration and card validation logic")
        if "authentication failed" in all_text:
            fix_suggestions.append("Review authentication flow and credential validation")
        if "database connection timeout" in all_text:
            fix_suggestions.append("Check database connection pool settings and network latency")
        if "disk" in all_text and "failed" in all_text:
            fix_suggestions.append("Investigate disk health and storage capacity")

        fix_suggestion = "; ".join(fix_suggestions) if fix_suggestions else "Review logs for specific error patterns and apply relevant fixes"

        return IncidentReport(
            incident_id=incident_id,
            summary=f"Incident analysis for: {query.question}. Found {len(evidence)} relevant log chunks across {len(affected_services)} services.",
            root_cause=root_cause,
            confidence=confidence,
            affected_services=affected_services,
            contributing_factors=factors if factors else ["See evidence for details"],
            timeline=timeline,
            evidence=evidence,
            fix_suggestion=fix_suggestion,
            requires_immediate_action="error" in all_levels or "critical" in all_levels,
        )

    def _build_prompt(self, query: QueryRequest, evidence: List[RetrievedChunk]) -> str:
        """Build the prompt from query and evidence."""
        parts = [
            f"User Question: {query.question}",
            "",
            "EVIDENCE (log chunks retrieved):",
            "=" * 60,
        ]

        for i, chunk in enumerate(evidence, 1):
            parts.append(f"\n--- Evidence Chunk {i} (score: {chunk.score:.2f}) ---")
            parts.append(f"Source: {', '.join(chunk.chunk.source_files)}")
            parts.append(f"Severity: {chunk.chunk.severity}")
            parts.append(f"Services: {', '.join(chunk.chunk.services)}")
            if chunk.chunk.time_range:
                parts.append(f"Time range: {chunk.chunk.time_range[0]} to {chunk.chunk.time_range[1]}")
            parts.append("\nLog content:")
            parts.append(chunk.chunk.text)
            parts.append("-" * 60)

        parts.append("")
        parts.append("INSTRUCTIONS:")
        parts.append("1. Identify the root cause based ONLY on the evidence above.")
        parts.append("2. List contributing factors.")
        parts.append("3. List affected services.")
        parts.append("4. Build a timeline of key events from the log timestamps.")
        parts.append("5. Provide a fix suggestion if the evidence supports one.")
        parts.append("6. Set confidence (0.0-1.0) based on how well the evidence supports your conclusion.")
        parts.append("7. Set requires_immediate_action=true if this is a critical error affecting users.")
        parts.append("8. Include the evidence chunks that support each conclusion.")

        return "\n".join(parts)