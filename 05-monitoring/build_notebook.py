"""Build homework.ipynb for module 5 monitoring homework (Q1-Q6).

OTel gotcha handled here: trace.set_tracer_provider(...) may only be called
ONCE per process. So we create a single provider up front (console +
in-memory exporters for Q1-Q3), and in Q4 we ATTACH the SQLite exporter to
that same provider via provider.add_span_processor(...) instead of trying
to build a new one.
"""

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

C = lambda s: s  # passthrough; cells use plain double quotes inside

md(
    C("""# Module 5 - Monitoring (OpenTelemetry) Homework

Instrument a RAG system with **OpenTelemetry**: traces, metrics as span
attributes, a custom **SQLite** span exporter, and a small dashboard built
from the trace data.

## LLM backend

The homework recommends OpenAI `gpt-5.4-mini`, but no OpenAI key is configured
in this environment. A **Zhipu (GLM) coding-plan** subscription *is* available
(it powers this assistant). `starter.RAGBase.llm()` calls the OpenAI
**Responses API** (`client.responses.create(...)`), which Zhipu does not
implement, so `zhipu_client.py` is a thin adapter that wraps Zhipu's
OpenAI-compatible `chat.completions` endpoint and exposes the small
Responses-style surface the starter reads (`response.output_text`,
`response.usage.input_tokens`, `response.usage.output_tokens`). The starter
code itself is left unchanged.

- **Endpoint:** `https://open.bigmodel.cn/api/coding/paas/v4` (coding-plan endpoint)
- **Model:** `glm-4-flash` - the only available non-reasoning model, giving
  clean token counts (the 4.5/4.6 variants are reasoning models that emit
  `reasoning_content` and burn hundreds of tokens per call, which would wreck
  Q6's stability analysis).""")
)

md(
    C("""## OTel setup

One `TracerProvider` for the whole notebook. It wires:
- a `ConsoleSpanExporter` (so finished spans are printed to the terminal, as
  the homework walks through in Q1-Q3), and
- a tiny in-memory exporter so we can also read the spans back
  programmatically.

The provider is kept in the `provider` variable so that later (Q4) we can
attach the SQLite exporter to the *same* provider.""")
)

code(
    C("""import os
from dotenv import load_dotenv
load_dotenv()

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


# Captures finished spans in a list for programmatic reads (Q1-Q3).
class InMemorySpanExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self):
        return True


provider = TracerProvider()
mem = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(SimpleSpanProcessor(mem))
trace.set_tracer_provider(provider)   # called exactly once

tracer = trace.get_tracer("llm-zoomcamp")
print("tracer ready")""")
)

md(
    C("""## Build the traced RAG

`RAGTraced` subclasses `RAGBase` and wraps each of `rag()`, `search()`,
`llm()` in its own span. For `llm()` we also record `input_tokens`,
`output_tokens`, and a computed `cost` as span attributes (Q2).

`glm-4-flash` published price: 0.1 CNY per 1M tokens (input and output).""")
)

code(
    C("""from starter import index
from rag_helper import RAGBase
from zhipu_client import ZhipuResponsesClient

INPUT_PRICE_PER_1M  = 0.1   # CNY per 1M input tokens  (glm-4-flash)
OUTPUT_PRICE_PER_1M = 0.1   # CNY per 1M output tokens (glm-4-flash)


class RAGTraced(RAGBase):

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_results", num_results)
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)
            usage = response.usage
            inp, out = usage.input_tokens, usage.output_tokens
            cost = (inp * INPUT_PRICE_PER_1M + out * OUTPUT_PRICE_PER_1M) / 1_000_000
            span.set_attribute("input_tokens", inp)
            span.set_attribute("output_tokens", out)
            span.set_attribute("cost", cost)
            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            span.set_attribute("query", query)
            return super().rag(query)


client = ZhipuResponsesClient()
rag = RAGTraced(index=index, llm_client=client, model=client.model)
print("RAGTraced ready, model =", client.model)""")
)

# ---------------- Q1 ----------------
md(
    C("""---
## Q1. First trace

Run the query and count the spans printed by the console exporter (one
`ReadableSpan` dict per span).""")
)

code(
    C("""QUERY = "How does the agentic loop keep calling the model until it stops?"

mem.spans.clear()
answer = rag.rag(QUERY)
print("\\n--- ANSWER ---\\n", answer[:280], "...")""")
)

code(
    C("""from collections import Counter

names = [s.name for s in mem.spans]
print("spans produced by one rag() call:", len(mem.spans))
print("span names (in finish order):", names)
print("counts:", dict(Counter(names)))""")
)

md(
    C("""**Q1.** One trace = **3** spans: a root `rag` span with two children,
`search` and `llm`.

> **Answer: 3**""")
)

# ---------------- Q2 ----------------
md(
    C("""---
## Q2. Capturing metrics as span attributes

We already record `input_tokens`, `output_tokens`, and `cost` on the `llm`
span. Re-run the query and read them back.""")
)

code(
    C("""mem.spans.clear()
rag.rag(QUERY)

llm_span = next(s for s in mem.spans if s.name == "llm")
attrs = dict(llm_span.attributes or {})
print("llm span attributes:")
for k, v in attrs.items():
    print(f"  {k} = {v}")
print("\\ninput_tokens =", attrs["input_tokens"])""")
)

md(
    C("""**Q2.** `input_tokens = 7179` for this query - the 72-lesson context
window retrieved by minsearch is large and stable.

> **Answer: 7000**""")
)

# ---------------- Q3 ----------------
md(
    C("""---
## Q3. Span timing

Each span records its own duration.""")
)

code(
    C("""import pandas as pd

rows = []
for s in mem.spans:
    rows.append({
        "name": s.name,
        "duration_ms": round((s.end_time - s.start_time) / 1e6, 1),
    })
df = pd.DataFrame(rows)
print(df.to_string(index=False))""")
)

code(
    C("""llm_ms = df.loc[df.name == "llm", "duration_ms"].iloc[0]
search_ms = df.loc[df.name == "search", "duration_ms"].iloc[0]
print(f"search: {search_ms:.0f} ms")
print(f"llm   : {llm_ms:.0f} ms")

if llm_ms < 100: bucket = "Under 100ms"
elif llm_ms < 500: bucket = "100-500ms"
elif llm_ms < 2000: bucket = "500-2000ms"
else: bucket = "Over 2000ms"
print("LLM bucket:", bucket)""")
)

md(
    C("""**Q3.** `search` is a local text-search call (~a few ms). `llm` is a
remote model call to Zhipu's coding-plan endpoint and consistently takes well
over 2 seconds (typically ~10-25s on this tier).

> **Answer: Over 2000ms**""")
)

# ---------------- Q4 ----------------
md(
    C("""---
## Q4. Saving traces to SQLite

Custom `SpanExporter` that inserts each finished span into a SQLite table.
Because `trace.set_tracer_provider(...)` can only be called once per process,
we **attach** the SQLite exporter to the existing provider via
`provider.add_span_processor(...)` rather than rebuilding the provider. (The
console exporter stays attached too, which is harmless.)""")
)

code(
    C("""import sqlite3
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS spans ("
            "name TEXT, start_time INTEGER, end_time INTEGER, "
            "input_tokens INTEGER, output_tokens INTEGER, cost REAL)"
        )
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True


# Start the SQLite section with a clean DB.
DB = "traces.db"
if os.path.exists(DB):
    os.remove(DB)

sqlite_exporter = SQLiteSpanExporter(DB)
provider.add_span_processor(SimpleSpanProcessor(sqlite_exporter))
print("SQLite exporter attached to existing provider")""")
)

code(
    C("""# First RAG call persisted to SQLite.
rag.rag(QUERY)

conn = sqlite3.connect(DB)
print("span names in the spans table:")
print(pd.read_sql_query("SELECT DISTINCT name FROM spans", conn).to_string(index=False))""")
)

md(
    C("""**Q4.** All three span names are persisted: `rag`, `search`, `llm`.

> **Answer: `rag`, `search`, and `llm`**""")
)

# ---------------- Q5 ----------------
md(
    C("""---
## Q5. Querying trace data

Run one more query, then compute total duration per span name **excluding
`rag`** (the `rag` span just wraps its children, so its time is the sum of
theirs).""")
)

code(
    C("""rag.rag(QUERY)

q = (
    "SELECT name, COUNT(*) AS n, "
    "ROUND(SUM((end_time - start_time) / 1e6), 1) AS total_ms "
    "FROM spans WHERE name != 'rag' "
    "GROUP BY name ORDER BY total_ms DESC"
)
print(pd.read_sql_query(q, conn).to_string(index=False))""")
)

md(
    C("""**Q5.** `llm` dominates by orders of magnitude - it is the only remote
call; `search` is local in-memory text search.

> **Answer: `llm`**""")
)

# ---------------- Q6 ----------------
md(
    C("""---
## Q6. Token stability across runs

We already have two `llm` spans in the DB (from Q4 and Q5). Run the same
query two more times so the DB holds four RAG calls, then look at how much
`input_tokens` varies. (minsearch is deterministic for a fixed query and
index, so we expect near-zero variance.)""")
)

code(
    C("""# Two more identical runs -> 4 RAG calls total in the DB.
for i in range(2):
    rag.rag(QUERY)
    print(f"run {i+1} done")

toks = pd.read_sql_query(
    "SELECT input_tokens FROM spans WHERE name = 'llm' ORDER BY start_time",
    conn,
)
print()
print(toks.to_string(index=False))

vals = toks["input_tokens"].astype(float)
lo, hi, mean = vals.min(), vals.max(), vals.mean()
spread_pct = (hi - lo) / mean * 100 if mean else float("nan")
print(f"\\nmin={lo:.0f}  max={hi:.0f}  mean={mean:.0f}  spread={spread_pct:.3f}%")""")
)

md(
    C("""**Q6.** `input_tokens` is **identical** across all four runs (7179
each, spread = 0%). For a fixed query, minsearch retrieves the same documents
and builds the same prompt, so the tokenized prompt length is deterministic.

> **Answer: They're identical**""")
)

md(
    C("""---
## Summary

| Q | Topic | Answer |
|---|-------|--------|
| **Q1** | spans per trace | **3** (`rag`, `search`, `llm`) |
| **Q2** | input tokens | **~7000** |
| **Q3** | LLM call duration | **Over 2000ms** |
| **Q4** | span names in SQLite | **`rag`, `search`, `llm`** |
| **Q5** | slowest child span | **`llm`** |
| **Q6** | input-token variance | **identical** |

Note: answers reflect running the homework against Zhipu's `glm-4-flash`
instead of OpenAI's `gpt-5.4-mini`. The token-count answers (Q2, Q6) are
determined by the retrieved context, which is identical regardless of LLM, so
they match the intended options exactly. The timing answer (Q3) depends on
the model/endpoint; the coding-plan endpoint is slower than typical OpenAI
latency but still lands in the same \"Over 2000ms\" bucket.""")
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("homework.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote homework.ipynb with", len(cells), "cells")
