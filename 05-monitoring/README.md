# Module 5 - Monitoring (OpenTelemetry)

Homework for LLM Zoomcamp module 5. Instruments the course-lessons RAG with
OpenTelemetry: traces, metrics as span attributes, a custom SQLite span
exporter, and a small dashboard built from the trace data.

## LLM backend

No OpenAI key is configured in this environment. The notebook uses a
**Zhipu (GLM) coding-plan** subscription instead. `zhipu_client.py` is a thin
adapter that wraps Zhipu's OpenAI-compatible `chat.completions` endpoint and
exposes the Responses-style API surface the starter reads
(`response.output_text`, `response.usage.input_tokens`,
`response.usage.output_tokens`), so `rag_helper.py` is left unchanged.

- Endpoint: `https://open.bigmodel.cn/api/coding/paas/v4`
- Model: `glm-4-flash` (non-reasoning; clean token counts)

## Setup

```bash
uv sync
# put credentials in .env:
#   ZHIPU_API_KEY=...
#   OPENAI_API_KEY=...   (optional; falls back to ZHIPU_API_KEY)
```

## Run

```bash
uv run jupyter notebook homework.ipynb
```

`build_notebook.py` regenerates `homework.ipynb` from source if needed.

## Answers

| Q | Topic | Answer |
|---|-------|--------|
| Q1 | spans per trace | 3 (`rag`, `search`, `llm`) |
| Q2 | input tokens | ~7000 (7179) |
| Q3 | LLM call duration | Over 2000ms |
| Q4 | span names in SQLite | `rag`, `search`, `llm` |
| Q5 | slowest child span | `llm` |
| Q6 | input-token variance | identical (spread 0%) |
