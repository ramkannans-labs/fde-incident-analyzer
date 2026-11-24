"""Data models for incident log analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
from typing import Optional, List


class LogEntry(BaseModel):
    """A single parsed log entry."""

    line_number: int = Field(ge=1, description="Line number in source file")
    timestamp: Optional[datetime] = Field(default=None, description="Log timestamp if parseable")
    level: str = Field(default="unknown", description="Log level: INFO, WARN, ERROR, FATAL, DEBUG")
    service: str = Field(default="unknown", description="Service name if identifiable")
    message: str = Field(description="Raw log message content")
    raw: str = Field(description="Original unmodified log line")
    source_file: str = Field(description="Path to source log file")
    metadata: dict = Field(default_factory=dict, description="Additional extracted fields")


class LogChunk(BaseModel):
    """A grouped chunk of logs for retrieval (e.g., 5 consecutive error lines)."""

    chunk_id: str = Field(description="Unique identifier for this chunk")
    entries: List[LogEntry] = Field(min_length=1, description="Log entries in this chunk")
    time_range: Optional[tuple[str, str]] = Field(default=None, description="Start and end ISO timestamps")
    severity: str = Field(default="info", description="info, warning, error, critical")
    services: List[str] = Field(default_factory=list, description="Services mentioned in chunk")
    source_files: List[str] = Field(default_factory=list, description="Files this chunk came from")
    text: str = Field(description="Combined text of chunk for embedding/search")


class RetrievedChunk(BaseModel):
    """A chunk retrieved during query processing with scoring."""

    chunk: LogChunk = Field(description="The retrieved log chunk")
    score: float = Field(ge=0.0, le=1.0, description="Retrieval score (0-1)")
    match_type: str = Field(description="keyword, semantic, hybrid")


class IncidentReport(BaseModel):
    """Structured incident report — the agent's final output."""

    incident_id: str = Field(description="Generated incident ID", examples=["INC-2024-12345"])
    summary: str = Field(description="One-sentence summary of the incident")
    root_cause: str = Field(description="Likely root cause")
    contributing_factors: List[str] = Field(default_factory=list, description="Secondary factors")
    affected_services: List[str] = Field(default_factory=list, description="Services impacted")
    timeline: List[dict] = Field(default_factory=list, description="Chronological event list")
    evidence: List[RetrievedChunk] = Field(default_factory=list, description="Supporting log chunks")
    fix_suggestion: Optional[str] = Field(default=None, description="Recommended remediation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in analysis (0-1)")
    requires_immediate_action: bool = Field(default=False, description="Needs urgent attention")


class QueryRequest(BaseModel):
    """User query about incident logs."""

    question: str = Field(description="Natural language question about logs")
    time_window_minutes: Optional[int] = Field(default=60, description="Search window in minutes")
    services: Optional[List[str]] = Field(default=None, description="Filter to specific services")
    severity_min: str = Field(default="error", description="Minimum severity: info, warning, error, critical")


class QueryResponse(BaseModel):
    """Response to user query."""

    report: IncidentReport = Field(description="Structured incident analysis")
    retrieved_count: int = Field(ge=0, description="Number of chunks retrieved")
    processing_time_ms: int = Field(ge=0, description="Total processing time")