"""Log parsing utilities for multiple log formats."""

from __future__ import annotations

import re
import json
from pathlib import Path
from datetime import datetime
from typing import Iterator, Optional
from ..schemas import LogEntry, LogChunk


# Common log patterns
LOG_PATTERNS = {
    "json": re.compile(r"^(\{.*\})$"),
    "common": re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[,\.]\d+)"
        r"\s+(?P<level>\w+)\s+"
        r"(?:\[(?P<service>[^\]]+)\]\s*)?"
        r"(?P<message>.*)$"
    ),
    "syslog": re.compile(
        r"^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
        r"\s+(?P<host>\S+)\s+"
        r"(?P<service>\S+?)(?:\[(?P<pid>\d+)\])?\s*:\s*"
        r"(?P<message>.*)$"
    ),
    "k8s": re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)"
        r"\s+(?P<stream>stdout|stderr)"
        r"\s+(?P<level>\w+)\s+"
        r"(?P<message>.*)$"
    ),
}

LEVEL_NORMALIZE = {
    "debug": "debug",
    "info": "info",
    "warn": "warning",
    "warning": "warning",
    "error": "error",
    "fatal": "critical",
    "critical": "critical",
    "err": "error",
}


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Try multiple timestamp formats."""
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S,%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%b %d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def normalize_level(level: str) -> str:
    """Normalize log level to standard."""
    return LEVEL_NORMALIZE.get(level.lower(), "info")


def extract_service(message: str) -> Optional[str]:
    """Try to extract service name from message."""
    # Common patterns: service-name: message, [service-name] message
    patterns = [
        r"^\[([a-zA-Z0-9-_]+)\]",
        r"^([a-zA-Z0-9-_]+):\s",
        r"service[=:]\s*([a-zA-Z0-9-_]+)",
    ]
    for pat in patterns:
        m = re.search(pat, message)
        if m:
            return m.group(1)
    return None


def parse_line(line: str, line_num: int, source_file: str) -> LogEntry:
    """Parse a single log line into LogEntry."""
    line = line.rstrip("\n")
    if not line:
        return LogEntry(
            line_number=line_num,
            message="",
            raw=line,
            source_file=source_file,
            level="empty",
        )

    # Try JSON first
    json_match = LOG_PATTERNS["json"].match(line)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return LogEntry(
                line_number=line_num,
                timestamp=parse_timestamp(data.get("timestamp", data.get("@timestamp", ""))),
                level=normalize_level(data.get("level", data.get("severity", "info"))),
                service=data.get("service", data.get("logger", "unknown")),
                message=data.get("message", data.get("msg", json.dumps(data))),
                raw=line,
                source_file=source_file,
                metadata={k: v for k, v in data.items() if k not in ["timestamp", "@timestamp", "level", "severity", "service", "logger", "message", "msg"]},
            )
        except json.JSONDecodeError:
            pass

    # Try common structured format
    match = LOG_PATTERNS["common"].match(line)
    if match:
        gd = match.groupdict()
        return LogEntry(
            line_number=line_num,
            timestamp=parse_timestamp(gd.get("timestamp", "")),
            level=normalize_level(gd.get("level", "info")),
            service=gd.get("service") or extract_service(gd.get("message", "")) or "unknown",
            message=gd.get("message", ""),
            raw=line,
            source_file=source_file,
        )

    # Try syslog format
    match = LOG_PATTERNS["syslog"].match(line)
    if match:
        gd = match.groupdict()
        return LogEntry(
            line_number=line_num,
            timestamp=parse_timestamp(gd.get("timestamp", "")),
            level="info",
            service=gd.get("service", "unknown"),
            message=gd.get("message", ""),
            raw=line,
            source_file=source_file,
        )

    # Try k8s format
    match = LOG_PATTERNS["k8s"].match(line)
    if match:
        gd = match.groupdict()
        return LogEntry(
            line_number=line_num,
            timestamp=parse_timestamp(gd.get("timestamp", "")),
            level=normalize_level(gd.get("level", "info")),
            service="k8s-container",
            message=gd.get("message", ""),
            raw=line,
            source_file=source_file,
        )

    # Fallback: treat as raw message
    return LogEntry(
        line_number=line_num,
        level="info",
        service=extract_service(line) or "unknown",
        message=line,
        raw=line,
        source_file=source_file,
    )


def parse_log_file(file_path: Path) -> Iterator[LogEntry]:
    """Parse a log file line by line, with special handling for JSON files."""
    # Skip Python script files that generate logs
    if file_path.suffix == ".py" or file_path.name == "sample.log":
        return

    # Try to parse as JSON first (for JSON log files that may be pretty-printed)
    content = file_path.read_text(encoding="utf-8", errors="replace")
    if file_path.suffix == ".json" or file_path.name.endswith(".json.log"):
        # Try to parse entire file as JSON
        try:
            data = json.loads(content)
            yield LogEntry(
                line_number=1,
                timestamp=parse_timestamp(data.get("timestamp", data.get("@timestamp", ""))),
                level=normalize_level(data.get("level", data.get("severity", "info"))),
                service=data.get("service", data.get("logger", "unknown")),
                message=data.get("message", data.get("msg", json.dumps(data))),
                raw=content,
                source_file=str(file_path),
                metadata={k: v for k, v in data.items() if k not in ["timestamp", "@timestamp", "level", "severity", "service", "logger", "message", "msg"]},
            )
            return
        except json.JSONDecodeError:
            pass
        # If not single JSON object, try NDJSON (one JSON per line)
        for i, line in enumerate(content.strip().split("\n"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield LogEntry(
                    line_number=i,
                    timestamp=parse_timestamp(data.get("timestamp", data.get("@timestamp", ""))),
                    level=normalize_level(data.get("level", data.get("severity", "info"))),
                    service=data.get("service", data.get("logger", "unknown")),
                    message=data.get("message", data.get("msg", json.dumps(data))),
                    raw=line,
                    source_file=str(file_path),
                    metadata={k: v for k, v in data.items() if k not in ["timestamp", "@timestamp", "level", "severity", "service", "logger", "message", "msg"]},
                )
            except json.JSONDecodeError:
                # Fall through to line-by-line parsing
                pass

    # Default: parse line by line
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            yield parse_line(line, i, str(file_path))


def chunk_logs(
    entries: list[LogEntry],
    max_chunk_size: int = 20,
    max_time_gap_seconds: int = 300,
) -> list[LogChunk]:
    """Group log entries into chunks by time proximity, severity, and source file."""
    if not entries:
        return []

    # First, group by source file to keep logs from different files separate
    from collections import defaultdict
    entries_by_file = defaultdict(list)
    for entry in entries:
        entries_by_file[entry.source_file].append(entry)

    all_chunks = []
    for source_file, file_entries in entries_by_file.items():
        chunks = []
        current_chunk: list[LogEntry] = []

        for entry in file_entries:
            if not current_chunk:
                current_chunk.append(entry)
                continue

            # Check time gap
            time_gap = None
            if current_chunk[-1].timestamp and entry.timestamp:
                time_gap = (entry.timestamp - current_chunk[-1].timestamp).total_seconds()

            # Check if we should start a new chunk
            should_split = (
                len(current_chunk) >= max_chunk_size
                or (time_gap is not None and time_gap > max_time_gap_seconds)
                or (current_chunk[-1].level in ["error", "critical"] and entry.level in ["error", "critical"] and len(current_chunk) >= 5)
            )

            if should_split:
                chunks.append(_build_chunk(current_chunk))
                current_chunk = [entry]
            else:
                current_chunk.append(entry)

        if current_chunk:
            chunks.append(_build_chunk(current_chunk))
        all_chunks.extend(chunks)

    return all_chunks


def _build_chunk(entries: list[LogEntry]) -> LogChunk:
    """Build a LogChunk from entries."""
    services = list({e.service for e in entries if e.service != "unknown"})
    source_files = list({e.source_file for e in entries})

    # Determine severity
    severity_order = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}
    max_severity = max((severity_order.get(e.level, 0) for e in entries), default=0)
    severity = {v: k for k, v in severity_order.items()}[max_severity]

    # Time range
    timestamps = [e.timestamp for e in entries if e.timestamp]
    time_range = None
    if timestamps:
        time_range = (min(timestamps).isoformat(), max(timestamps).isoformat())

    # Combined text - include source file name for better retrieval
    source_name = Path(source_files[0]).stem if source_files else "unknown"
    text = "\n".join(f"[{e.timestamp or 'NO_TIME'}] [{e.level}] [{e.service}] {e.message}" for e in entries)
    text = f"[source: {source_name}]\n{text}"

    # Generate unique chunk ID using source file hash
    import hashlib
    file_hash = hashlib.md5(source_files[0].encode()).hexdigest()[:8]
    chunk_id = f"chunk_{file_hash}_{entries[0].line_number}_{entries[-1].line_number}"

    return LogChunk(
        chunk_id=chunk_id,
        entries=entries,
        time_range=time_range,
        severity=severity,
        services=services,
        source_files=source_files,
        text=text,
    )