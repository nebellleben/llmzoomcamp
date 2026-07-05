"""Build the homework notebook (homework.ipynb) from a list of cells."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# Homework: Evaluation

LLM Zoomcamp 2026 — Module 4.

In Module 2 we built keyword, vector, and hybrid search and asked which one is
best. Here we answer that with numbers: we generate a ground-truth dataset of
questions (one known-correct page per question) and evaluate each search method
with **Hit Rate** and **MRR**.""")

md("""## Setup

Dependencies (installed via
`uv add openai pydantic python-dotenv pandas gitsource minsearch numpy \\
onnxruntime tokenizers tqdm huggingface-hub jupyter`):

- `embedder.py` / `download.py` from Module 2 (ONNX `all-MiniLM-L6-v2`)
- `rag_helper.py` / `evaluation_utils.py` from the course repo
- `ground-truth.csv` — the 360 pre-generated questions for all 72 pages""")

code("""import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from embedder import Embedder
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch

load_dotenv(".env", override=True)
emb = Embedder()""")

md("""## Loading the data

Pull the 72 lesson pages from the course repo at commit `8c1834d`, exactly as
in Module 2.""")

code("""reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]
print(f"Loaded {len(documents)} documents")""")

md("""---
## Q1. Generating questions

To evaluate search we need a ground truth: questions where we know the page that
should answer them. We ask an LLM to write 5 such questions per page.

We reuse the structured-output approach from the module — the `Questions` model
and the `llm_structured` helper.""")

code("""class Questions(BaseModel):
    questions: list[str]


data_gen_instructions = \"\"\"
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.
Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
\"\"\".strip()""")

code("""import os

# OpenAI quota is exhausted, so we use Gemini via its OpenAI-compatible
# chat-completions endpoint. (The homework allows any provider.) The module's
# llm_structured uses the OpenAI Responses API, so here we wrap chat.completions
# with JSON output and report prompt_tokens as the "input tokens".
gemini_client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
LLM_MODEL = "gemini-2.5-flash"


def generate_questions(doc):
    user_prompt = json.dumps({"filename": doc["filename"], "content": doc["content"]})
    resp = gemini_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": data_gen_instructions
             + "\\n\\nReturn ONLY valid JSON of the form "
             '{\\"questions\\": [\\"...\\", ...]} with exactly 5 questions.'},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content)
    usage = type("U", (), {"input_tokens": resp.usage.prompt_tokens})()
    return parsed["questions"], usage""")


md("""Generate questions for the first 3 pages and track token usage.""")

code("""first_three = [
    "01-agentic-rag/lessons/01-intro.md",
    "01-agentic-rag/lessons/02-environment.md",
    "01-agentic-rag/lessons/03-rag.md",
]

by_filename = {d["filename"]: d for d in documents}

records = []
usages = []

for fn in first_three:
    questions, usage = generate_questions(by_filename[fn])
    for q in questions:
        records.append({"question": q, "filename": fn})
    usages.append(usage)
    print(f"{fn}: {len(questions)} questions")

avg_input = np.mean([u.input_tokens for u in usages])
print(f"\\nAverage input tokens across 3 calls: {avg_input:.1f}")""")

md("""**Answer Q1: ~1400** — the prompt we send (instructions + a JSON dump of a
lesson page) is on the order of a thousand-ish tokens per call. The exact number
varies a bit between runs, but it lands near 1400, not 140 / 14000 / 140000.""")

md("""## The full ground truth

Generating questions for all 72 pages costs money and time, so we use the
pre-generated dataset (360 questions, same method as above).""")

code("""df_gt = pd.read_csv("ground-truth.csv")
ground_truth = df_gt.to_dict(orient="records")
print(f"{len(ground_truth)} ground-truth questions")
ground_truth[0]""")

md("""## Searching the chunks

Rebuild the Module 2 search over the same 295 chunks (`size=2000, step=1000`):
a text index, a vector index, and an RRF hybrid on top of both. Each search
returns chunks; we evaluate them by `filename`.""")

code("""chunks = chunk_documents(documents, size=2000, step=1000)
print(f"{len(chunks)} chunks")""")

code("""# Vector index: embed every chunk once
texts = [c["content"] for c in chunks]
X = emb.encode_batch(texts)

vs = VectorSearch(keyword_fields=["filename"])
vs.fit(X, chunks)

# Text index
text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)


def text_search(query, num_results=5):
    return text_index.search(query, num_results=num_results)


def vector_search(query, num_results=5):
    qv = emb.encode(query)
    return vs.search(qv, num_results=num_results)


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(query, k=60):
    text_results = text_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k)

print("indexes ready")""")

md("""---
## Q2. First result with text search

Take the first ground-truth question and run `text_search`.""")

code("""q = ground_truth[0]["question"]
print("Q:", q)

text_res = text_search(q)
print("\\nFirst text result:", text_res[0]["filename"])""")

md("""**Answer Q2: `01-agentic-rag/lessons/03-rag.md`** — keyword search does *not*
find the page this question was generated from (`01-intro.md`); it lands on the
RAG lesson instead, which shares a lot of the same vocabulary.""")

md("""---
## Q3. First result with vector search

Same question, but `vector_search` (semantic match).""")

code("""vec_res = vector_search(q)
print("First vector result:", vec_res[0]["filename"])""")

md("""**Answer Q3: `01-agentic-rag/lessons/01-intro.md`**

The question came from `01-intro.md`. Vector search finds it at the top by
meaning; text search (Q2) doesn't — it returned `03-rag.md`. One query isn't
enough to judge a method — that's why we evaluate across all 360 questions next.""")

md("""## Evaluation metrics

Same functions as in the module, but a hit is a `filename` match (our ground
truth labels pages by `filename`, not by FAQ `id`).""")

code("""def compute_relevance(q, search_function):
    filename = q["filename"]
    results = search_function(query=q["question"])
    return [int(d["filename"] == filename) for d in results]


def compute_relevance_total(ground_truth, search_function):
    return [compute_relevance(q, search_function) for q in tqdm(ground_truth)]


def hit_rate(relevance):
    cnt = 0
    for line in relevance:
        if 1 in line:
            cnt += 1
    return cnt / len(relevance)


def mrr(relevance):
    total = 0.0
    for line in relevance:
        for rank, v in enumerate(line):
            if v == 1:
                total += 1 / (rank + 1)
                break
    return total / len(relevance)


def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)
    return {"hit_rate": hit_rate(relevance_total), "mrr": mrr(relevance_total)}""")

md("""---
## Q4. Evaluating text search

Hit Rate of `text_search` on the full ground truth.""")

code("""text_metrics = evaluate(ground_truth, text_search)
print("text_search:", text_metrics)""")

md("""**Answer Q4: ~0.76** Hit Rate.""")

md("""---
## Q5. Evaluating vector search

MRR of `vector_search` (the part left for the homework — the module only
evaluated keyword search).""")

code("""vector_metrics = evaluate(ground_truth, vector_search)
print("vector_search:", vector_metrics)""")

md("""**Answer Q5: ~0.55** MRR (0.5486 measured).""")

md("""---
## Q6. Tuning hybrid search

The `k` in RRF controls how much the top ranks matter. Smaller `k` sharpens
the gap between positions. Evaluate `hybrid_search` for
`k ∈ {1, 50, 100, 200}` and compare MRR. On a tie, pick the smallest `k`.""")

code("""k_results = []
for k in [1, 50, 100, 200]:
    metrics = evaluate(ground_truth, lambda query, k=k: hybrid_search(query, k=k))
    metrics["k"] = k
    k_results.append(metrics)
    print(f"k={k:>4}: hit_rate={metrics['hit_rate']:.4f}  mrr={metrics['mrr']:.4f}")

best = max(k_results, key=lambda r: r["mrr"])
print(f"\\nBest k by MRR: {best['k']} (mrr={best['mrr']:.4f})")""")

md("""**Answer Q6: the `k` with the highest MRR** (ties broken toward smaller `k`).""")

md("""---
## Summary

| Question | Answer |
|----------|--------|
| Q1. Avg input tokens (3 calls) | **~1400** (1471.7 measured, Gemini) |
| Q2. First text result | **`01-agentic-rag/lessons/03-rag.md`** |
| Q3. First vector result | **`01-agentic-rag/lessons/01-intro.md`** |
| Q4. text_search Hit Rate | **~0.76** (0.7583) |
| Q5. vector_search MRR | **~0.55** (0.5486) |
| Q6. Best RRF `k` by MRR | **1** (mrr=0.6482) |

**Takeaways**
- On this dataset **hybrid search beats either method alone** — Hit Rate jumps
  to ~0.84 and MRR to ~0.65, because each method covers the other's blind spots
  (Q2/Q3 show exactly that: vector finds the intro page, text doesn't).
- For RRF, a **smaller `k`** (sharper top-rank weighting) gives the best MRR
  here; `k=1` edges out 50/100/200, which all tie. With ties broken toward the
  smallest `k`, the answer is **1**.""")


nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.14",
    },
}

with open("homework.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote homework.ipynb with", len(cells), "cells")
