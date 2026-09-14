"""
answer.py — Answer synthesis with Gemini 2.5 Flash + refusal guard.

Guardrails enforced here:
  1. Distance cutoff (MAX_DISTANCE = 0.5): if best retrieval distance > 0.5,
     abort immediately — zero LLM tokens spent.
  2. Strict system prompt: 3-4 sentence Hinglish, NO code, mandatory badge.
  3. Citation filtering: only return citations [N] explicitly referenced
     by the model in its response text.
  4. Hard output cap: max_output_tokens=350.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from google import genai
from google.genai import types as genai_types
from qdrant_client.http.models import ScoredPoint

import config
from models import Citation, ComplexityBadge, RAGResponse, parse_badge
from preprocess import preprocess_query


# ── System prompt (exact, immutable) ──────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert DSA mentor answering questions based on video lecture transcripts.

LANGUAGE & TONE:
- Primary Language: English with a light, natural touch of Hinglish (e.g., "basically", "dhyan do", "divide karte hain", "yaad rakho").
- Adaptable: Handle questions asked in English, Hinglish, or Hindi seamlessly with the same crisp style.

CRITICAL REQUIREMENT — BE CONCISE & TO THE POINT:
- Keep the entire explanation compact (around 3-4 concise bullet points or 1-2 tight paragraphs, max 100-140 words total).
- Avoid long introductory fluff, greetings, repetitive disclaimers, or full code blocks.
- STRICT FORMATTING:
  * NEVER output markdown headers (NO #, ##, ###).
  * NEVER output LaTeX blocks or dollar signs (NO $ or $$ or \text{}). Write formulas in plain clean text.
  * Use bold labels like **Core Intuition:** or **How it works:**.
  * Use single citation brackets like [1] or [2] (do not combine like [1, 2]).
- Explain:
  1. Core intuition (what and why).
  2. How it works in 2-3 brief logical steps.
  3. Inline citations [1], [2] referencing the transcript source for timestamps.

MANDATORY BADGE (exact line at the very end):
[Time: O(...) | Space: O(...) | Pattern: ...]
(Deduce standard time/space complexity if not explicitly mentioned).

OUT-OF-DOMAIN REFUSAL:
If the topic or question is NOT covered in the provided lecture transcripts, output ONLY:
"Ye topic in lectures me cover nahi hua."
"""

# ── Gemini client singleton ────────────────────────────────────────────────

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _gemini_client


# ── Context assembly ───────────────────────────────────────────────────────

def _build_context(points: list[ScoredPoint], max_chunks: int = 5, max_chars_per_chunk: int = 750) -> str:
    """
    Assemble numbered transcript blocks for the user turn.
    Uses top chunks with enough depth (750 chars each) to supply complete
    algorithmic context and accurate timestamps.
    """
    blocks: list[str] = []
    for i, pt in enumerate(points[:max_chunks], start=1):
        p = pt.payload or {}
        title = p.get("title", "Unknown Lecture")
        start = p.get("start_sec", 0)
        text = (p.get("text", ""))[:max_chars_per_chunk].strip()
        blocks.append(f"[TRANSCRIPT {i} — {title} @ {int(start)}s]\n{text}")
    return "\n\n".join(blocks)


# ── Citation extraction ────────────────────────────────────────────────────

_REF_RE = re.compile(r"\[(\d+)\]")


def _extract_cited_indices(answer_text: str, max_idx: int) -> list[int]:
    """
    Return 0-based indices of citations explicitly referenced in the answer.
    E.g., "[1] ... [3]" → [0, 2]
    """
    refs = {int(m) - 1 for m in _REF_RE.findall(answer_text)}
    return sorted(r for r in refs if 0 <= r < max_idx)


def _point_to_citation(pt: ScoredPoint, fallback_distance: float = 0.0) -> Citation:
    """Convert a Qdrant ScoredPoint to a Citation model."""
    p = pt.payload or {}
    text = p.get("text", "")
    preview = text[:120] + ("…" if len(text) > 120 else "")
    score = float(pt.score)
    # Cosine score is typically 0.0 - 1.0; RRF score is typically < 0.05
    if score > 0.05:
        distance = max(0.0, 1.0 - score)
    else:
        distance = fallback_distance

    return Citation(
        video_id=p.get("video_id", ""),
        title=p.get("title", ""),
        start_sec=float(p.get("start_sec", 0)),
        end_sec=float(p.get("end_sec", 0)),
        seek_sec=float(p.get("seek_sec", 0)),
        youtube_url=p.get("youtube_url", ""),
        chunk_preview=preview,
        distance=round(distance, 4),
        segments=p.get("segments", []),
    )


# ── Main entrypoint ────────────────────────────────────────────────────────

REFUSAL_MSG = "Ye topic in lectures me cover nahi hua."


def answer(
    query: str,
    points: list[ScoredPoint],
    retrieval_distance: float | None = None,
    query_processed: str = "",
    t0: float | None = None,
) -> RAGResponse:
    """
    Full answer synthesis pipeline with guardrails.

    Args:
        query:               Raw user query (for display in response).
        points:              Top-K ScoredPoints from Qdrant (ordered by score desc).
        retrieval_distance:  Cosine distance metric from dense match (if hybrid).
        query_processed:     Preprocessed query string (after acronym expansion).
        t0:                  Pipeline start time (for latency_ms).

    Returns:
        RAGResponse — always a valid response, even on refusal or error.
    """
    if t0 is None:
        t0 = time.perf_counter()

    if not query_processed:
        query_processed = preprocess_query(query)

    def _elapsed_ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    # ── Guard 1: no results at all ─────────────────────────────────────────
    if not points:
        return RAGResponse(
            answer=REFUSAL_MSG,
            is_refused=True,
            retrieval_distance=1.0,
            tokens_used=0,
            query_processed=query_processed,
            latency_ms=_elapsed_ms(),
        )

    # ── Guard 2: distance cutoff ───────────────────────────────────────────
    if retrieval_distance is not None:
        best_distance = float(retrieval_distance)
    else:
        top_score = float(points[0].score)
        # If top_score is tiny (<= 0.05, typical of RRF or non-match), distance is 1.0 (unmatched)
        best_distance = max(0.0, 1.0 - top_score) if top_score > 0.05 else 1.0

    if best_distance > config.MAX_DISTANCE:
        return RAGResponse(
            answer=REFUSAL_MSG,
            is_refused=True,
            retrieval_distance=round(best_distance, 4),
            tokens_used=0,
            query_processed=query_processed,
            latency_ms=_elapsed_ms(),
        )

    # ── Build context ──────────────────────────────────────────────────────
    context = _build_context(points)
    user_turn = (
        f"Lecture Transcripts:\n\n{context}\n\n"
        f"Question: {query_processed or query}"
    )

    # ── Call Gemini with Retry & Fallback ──────────────────────────────────
    client = _get_gemini_client()

    max_retries = 3
    last_err: Exception | None = None
    response = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=user_turn)],
                    )
                ],
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=config.TEMPERATURE,
                    max_output_tokens=config.MAX_OUTPUT_TOKENS,
                    top_p=config.TOP_P,
                    top_k=config.TOP_K_GEMINI,
                    candidate_count=1,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            break
        except Exception as exc:
            last_err = exc
            err_msg = str(exc)
            # Retry on transient server errors (503 / 429)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries - 1:
                wait_sec = 2 ** attempt
                print(f"[answer] Gemini transient error ({exc}), retrying in {wait_sec}s (attempt {attempt + 1}/{max_retries}) …")
                time.sleep(wait_sec)
                continue
            break

    if response is None:
        # Fallback gracefully instead of crashing with unhandled ServerError
        err_detail = "Server temporarily busy. Please try again in a few seconds."
        print(f"[answer] Gemini call failed after retries: {last_err}")
        return RAGResponse(
            answer=f"⚠️ {err_detail}",
            complexity_badge=None,
            citations=[],
            retrieval_distance=round(best_distance, 4),
            tokens_used=0,
            is_refused=False,
            query_processed=query_processed or query,
            latency_ms=_elapsed_ms(),
        )

    answer_text: str = response.text or REFUSAL_MSG

    # Token accounting
    usage = response.usage_metadata
    tokens_used: int = 0
    if usage:
        tokens_used = (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)

    # ── Parse complexity badge ─────────────────────────────────────────────
    badge: ComplexityBadge | None = parse_badge(answer_text)

    # ── Filter to explicitly cited chunks only ─────────────────────────────
    cited_indices = _extract_cited_indices(answer_text, len(points))
    # If model produced no [N] refs at all, surface the top result anyway
    if not cited_indices:
        cited_indices = [0]

    citations = [_point_to_citation(points[i], fallback_distance=best_distance) for i in cited_indices]

    return RAGResponse(
        answer=answer_text,
        complexity_badge=badge,
        citations=citations,
        retrieval_distance=round(best_distance, 4),
        tokens_used=tokens_used,
        is_refused=False,
        query_processed=query_processed or query,
        latency_ms=_elapsed_ms(),
    )
