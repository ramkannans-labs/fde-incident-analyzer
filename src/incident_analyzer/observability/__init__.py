"""Observability for incident-analyzer — OpenTelemetry tracing + Langfuse export."""

from __future__ import annotations

from typing import Optional

from ..config import get_settings


def setup_tracing() -> None:
    """Configure OpenTelemetry tracing for incident-analyzer.

    If OTEL_EXPORTER_OTLP_ENDPOINT is set, exports to Langfuse.
    Otherwise, no-op (local-only).
    """
    settings = get_settings()

    # Check if OTEL endpoint is configured
    otlp_endpoint = settings.otlp_endpoint
    if not otlp_endpoint:
        # No observability endpoint configured - skip tracing setup
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        resource = Resource.create({
            "service.name": "incident-analyzer",
            "service.version": "0.1.0",
        })

        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
    except Exception as e:
        # Silently fail if tracing setup fails
        print(f"Warning: Tracing setup failed: {e}")


def get_tracer(name: str = "incident-analyzer"):
    """Get a tracer for the given name."""
    from opentelemetry import trace
    return trace.get_tracer(name)


def record_incident_analysis(
    incident_id: str,
    question: str,
    retrieved_count: int,
    processing_time_ms: int,
    confidence: float,
) -> None:
    """Record a metric event for an incident analysis."""
    try:
        from opentelemetry import metrics
        meter = metrics.get_meter("incident-analyzer")

        # Record analysis count
        analysis_counter = meter.create_counter(
            name="incident.analyses",
            description="Number of incident analyses performed",
        )
        analysis_counter.add(1, {
            "incident_id": incident_id,
            "confidence": confidence,
        })
    except Exception:
        # Silently fail if metrics setup fails
        pass