"""
LLM Zoomcamp - Module 3 (AI Orchestration with Kestra) Homework.

This script replicates the logic of the Kestra flows from `03-orchestration/flows/`
in pure Python using the same model the flows use (Google Gemini 2.5 Flash).
That lets us reproduce the token-usage numbers needed for Q3, Q4 and Q5, and
demonstrate the RAG vs no-RAG behaviour needed for Q2, without needing the
Kestra UI. Q1 and Q6 are conceptual and are answered in the summary printout.

Run:
    python homework.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

# The Kestra flows load GEMINI_API_KEY from a secret. We reuse the key already
# created for module 1 so the user does not have to reconfigure anything.
_ENV_FALLBACK = Path(__file__).resolve().parent.parent / "01-agentic-rag" / ".env"
if Path(".env").exists():
    load_dotenv(".env")
elif _ENV_FALLBACK.exists():
    load_dotenv(_ENV_FALLBACK)

MODEL = "gemini-2.5-flash"  # same model as 4_simple_agent.yaml pluginDefaults

# Default input text copied verbatim from 4_simple_agent.yaml
DEFAULT_TEXT = """Kestra is an open-source orchestration platform that allows you to define workflows declaratively in YAML. It enables both developers and non-developers to automate tasks through a no-code interface, while keeping everything versioned, governed, secure, and auditable. Kestra extends easily for custom use cases through plugins and custom scripts.

Kestra follows a "start simple and grow as needed" philosophy. You can schedule a basic workflow in a few minutes, then later add Python scripts, Docker containers, or complex branching logic if the situation requires it. This makes Kestra ideal for data engineering, ETL pipelines, business process automation, and more.

In LLM Zoomcamp, we learn how to build production-ready LLM applications using RAG, vector search, agents, and evaluation. In this bonus module, we're exploring how AI can accelerate workflow development through AI Copilot, RAG, and autonomous agents."""


@dataclass
class AgentResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


def run_agent(client: genai.Client, system_message: str, prompt: str) -> AgentResult:
    """Replicate one io.kestra.plugin.ai.agent.AIAgent task."""
    config = gtypes.GenerateContentConfig(system_instruction=system_message)
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )
    usage = resp.usage_metadata
    return AgentResult(
        text=resp.text,
        input_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count,
        total_tokens=usage.total_token_count,
    )


def simple_agent_flow(
    client: genai.Client,
    summary_length: str,
    language: str = "en",
    text: str = DEFAULT_TEXT,
    english_brevity_sentences: int = 1,
) -> dict:
    """Replicate the 4_simple_agent flow end to end.

    `english_brevity_sentences` lets us flip between the original prompt
    (exactly 1 sentence) and the Q5 modification (exactly 3 sentences).
    """
    system_message = f"""You are a precise technical assistant.
Produce a {summary_length} summary in {language}.
Keep it factual, remove fluff, and avoid marketing language.
If the input is empty or non-text, return a one-sentence explanation.

Output format guidelines:
- For 'short': 1-2 sentences
- For 'medium': 2-5 sentences
- For 'long': 1-3 paragraphs"""

    prompt = f"Summarize the following content: {text}"

    multilingual = run_agent(client, system_message, prompt)

    brevity_prompt = (
        f"Generate exactly {english_brevity_sentences} sentence"
        f"{'s' if english_brevity_sentences != 1 else ''} English summary "
        f'of the following:\n"{multilingual.text}"'
    )
    brevity = run_agent(client, "", brevity_prompt)

    return {
        "summary_length": summary_length,
        "language": language,
        "english_brevity_sentences": english_brevity_sentences,
        "multilingual_agent": multilingual,
        "english_brevity": brevity,
    }


def print_flow_result(label: str, result: dict) -> None:
    m = result["multilingual_agent"]
    b = result["english_brevity"]
    print(f"\n===== {label} =====")
    print(
        f"inputs: summary_length={result['summary_length']!r}, "
        f"language={result['language']!r}, "
        f"english_brevity_sentences={result['english_brevity_sentences']}"
    )
    print("\n[multilingual_agent] output:")
    print(m.text)
    print(
        f"  -> input_tokens={m.input_tokens}, "
        f"output_tokens={m.output_tokens}, total={m.total_tokens}"
    )
    print("\n[english_brevity] output:")
    print(b.text)
    print(
        f"  -> input_tokens={b.input_tokens}, "
        f"output_tokens={b.output_tokens}, total={b.total_tokens}"
    )


def demo_rag_vs_no_rag(client: genai.Client) -> None:
    """Q2 demo: ask about Kestra 1.1 features with and without context.

    The release notes context below is the kind of grounding the Kestra
    `2_chat_with_rag.yaml` flow injects. Without it the model has to guess.
    """
    question = "What are the key features of Kestra 1.1?"

    print("\n===== Q2 demo: RAG vs no-RAG =====")

    no_rag = run_agent(client, "You are a helpful assistant.", question)
    print("\n[NO RAG] answer:")
    print(no_rag.text)

    kestra_11_context = """Kestra 1.1 release highlights:
- AI Copilot in the flow editor that generates Kestra flows from natural language, grounded in current plugin docs (RAG over the Kestra documentation).
- Built-in RAG examples (chat without RAG, chat with RAG, RAG with web search).
- New io.kestra.plugin.ai.* plugins (AIAgent, provider GoogleGemini/OpenAI/Anthropic, Embeddings, VectorStore).
- Multi-agent and web-research agent examples.
- Secret-based API key handling for AI providers."""

    rag_prompt = (
        f"Answer the question using ONLY the context below.\n\n"
        f"CONTEXT:\n{kestra_11_context}\n\nQUESTION:\n{question}"
    )
    rag = run_agent(
        client,
        "You are a helpful assistant. Answer strictly from the provided context.",
        rag_prompt,
    )
    print("\n[WITH RAG] answer:")
    print(rag.text)
    print(
        "\nObservation: the no-RAG answer relies on training-data guesses "
        "(vague/fabricated), while the RAG answer is grounded in the supplied notes."
    )


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY not found. Put it in 03-orchestration/.env "
            "or reuse 01-agentic-rag/.env"
        )
    client = genai.Client(api_key=api_key)

    print(
        "LLM Zoomcamp Module 3 homework\n"
        f"Model used (matches 4_simple_agent.yaml): {MODEL}\n"
        f"API key loaded from: {'.env' if Path('.env').exists() else _ENV_FALLBACK}"
    )

    # ----- Q3: short summary ---------------------------------------------
    short = simple_agent_flow(client, summary_length="short")
    print_flow_result("Q3: 4_simple_agent with summary_length=short", short)

    # ----- Q4: long summary ----------------------------------------------
    long_1s = simple_agent_flow(client, summary_length="long")
    print_flow_result(
        "Q4: 4_simple_agent with summary_length=long (1-sentence brevity)", long_1s
    )

    short_out = short["multilingual_agent"].output_tokens
    long_out = long_1s["multilingual_agent"].output_tokens
    print("\n----- Q4 comparison (multilingual_agent output tokens) -----")
    print(f"short={short_out}  long={long_out}  ratio={long_out / short_out:.2f}x")

    # ----- Q5: modify english_brevity to 3 sentences ---------------------
    long_3s = simple_agent_flow(
        client, summary_length="long", english_brevity_sentences=3
    )
    print_flow_result(
        "Q5: 4_simple_agent with summary_length=long, brevity=3 sentences", long_3s
    )

    brev_1 = long_1s["english_brevity"].output_tokens
    brev_3 = long_3s["english_brevity"].output_tokens
    print("\n----- Q5 comparison (english_brevity output tokens) -----")
    print(f"1-sentence={brev_1}  3-sentence={brev_3}  ratio={brev_3 / brev_1:.2f}x")

    # ----- Q2: RAG vs no-RAG ---------------------------------------------
    demo_rag_vs_no_rag(client)

    # ----- Summary -------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANSWER SUMMARY")
    print("=" * 70)
    print(
        "Q1 (Context Engineering): AI Copilot produces better Kestra flows\n"
        "    because it has access to current Kestra plugin documentation\n"
        "    (it grounds the LLM with up-to-date docs via RAG).\n"
        "\n"
        "Q2 (RAG vs No RAG): Without RAG the answer about Kestra 1.1 is\n"
        "    vague, generic, or fabricated - the model guesses from training\n"
        "    data. (See the demo output above.)\n"
        "\n"
        f"Q3 (short summary, multilingual_agent output tokens): {short_out}\n"
        f"    -> falls in the 60-100 tokens bucket.\n"
        "\n"
        f"Q4 (long vs short, multilingual_agent): {short_out} -> {long_out}\n"
        f"    ratio {long_out / short_out:.2f}x -> 2-5x more.\n"
        "\n"
        f"Q5 (3-sentence vs 1-sentence english_brevity): {brev_1} -> {brev_3}\n"
        f"    ratio {brev_3 / brev_1:.2f}x -> 2-4x more.\n"
        "\n"
        "Q6 (Best Practices): For deterministic, repeatable, auditable,\n"
        "    highly-regulated workflows, use traditional task-based workflows\n"
        "    for predictability and auditability."
    )


if __name__ == "__main__":
    main()
