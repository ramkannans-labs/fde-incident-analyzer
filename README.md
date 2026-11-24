# Incident Analyzer — FDE Portfolio Project

A local-first incident log analyzer demonstrating Forward Deployed Engineer (FDE) skills through practical, token-efficient examples. Each component teaches a core concept (RAG, structured output, evaluation, observability) without high API costs.

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Start local infrastructure
docker-compose up -d

# 3. Copy .env.example and configure
cp .env.example .env

# 4. Index sample logs
python scripts/index.py

# 5. Ask a question
python scripts/ask.py "What caused the authentication failures?"

# 6. Evaluate against golden set
python scripts/eval.py tests/golden_set.jsonl
```

## Architecture

The project is structured as a proper Python package (`src/incident_analyzer/`) with the following modules:

| Module | Concept |
|--------|---------|
| `ingestion.parser` | Multi-format log parsing (JSON, common, syslog, k8s) |
| `retrieval` | BM25 keyword search + optional semantic reranking |
| `generation` | PydanticAI agent with structured `IncidentReport` output |
| `evaluation` | Golden set evaluation with groundedness/faithfulness metrics |
| `observability` | OpenTelemetry + Langfuse tracing (local Docker) |
| `schemas` | Pydantic data models (LogEntry, LogChunk, IncidentReport) |
| `config` | Environment-driven configuration via .env |

## Key Concepts Demonstrated

- **BM25 + Ollama embeddings**: Zero-API-cost local retrieval
- **Hybrid retrieval**: BM25 + optional semantic reranking
- **Multi-format log parsing**: JSON, common structured, syslog, k8s formats
- **PydanticAI structured output**: `IncidentReport` output_type with citations
- **Golden set evaluation**: Static test suite for groundedness, faithfulness, answer relevance
- **Policy-as-code**: Incident classification rules as Pydantic models
- **OpenTelemetry tracing**: End-to-end span tracking to local Langfuse
- **Qdrant vector database**: Local vector storage

## Commands

### `python scripts/index.py`
Index log chunks from sample data into the BM25 index and optionally Qdrant.

### `python scripts/ask.py "question"`
Ask a natural language question about incident logs. Generates a structured incident report with:
- Summary, root cause, contributing factors
- Affected services and confidence score
- Fix suggestion and immediate action flag
- Evidence citations with scores and match types

### `python scripts/eval.py tests/golden_set.jsonl`
Evaluate agent output against a golden set of expected answers. Reports per-entry metrics and aggregate pass rates.

### `python scripts/index.py --eval`
Run evaluation after indexing.

## Infrastructure

Run all local services with Docker Compose:

```bash
docker-compose up -d
```

Services:
- **qdrant**: Vector database at `localhost:6333`
- **ollama**: Local LLM embeddings at `localhost:11434`
- **langfuse**: Observability dashboard at `http://localhost:3000`

## License

MIT — free for learning and portfolio use.