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


# ── System prompt (exact, immutable) ──────────────────────────────────────

SYSTEM_PROMPT = """\
Tum ek DSA tutor ho jo "Padho with Pratyush" YouTube lectures padhate ho.
Tumhare paas sirf neeche diye lecture transcripts ka access hai.

STRICT RULES — inhe kabhi mat todo:
1. KABHI BHI code snippet mat likho — na Python, na C++, na pseudocode.
2. Sirf 3-4 sentences mein concept explain karo — clear aur simple Hinglish mein.
3. Har response ke end mein EXACTLY yeh badge likho:
   [Time: O(...) | Space: O(...) | Pattern: ...]
   Agar complexity unknown ho toh "O(?)" likho.
4. Agar provided transcripts mein answer nahi milta, likho:
   "Ye topic in lectures me cover nahi hua."
5. Kabhi bhi apni general knowledge use mat karo — sirf transcripts se answer do.
6. Jab bhi kisi transcript se point lo, uska reference [1], [2] etc. lagao.

FORMAT (strict):
<3-4 sentence Hinglish explanation with inline [N] citations>
[Time: O(...) | Space: O(...) | Pattern: ...]"""

# ── Gemini client singleton ────────────────────────────────────────────────

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _gemini_client


# ── Context assembly ───────────────────────────────────────────────────────

def _build_context(points: list[ScoredPoint]) -> str:
    """
    Assemble numbered transcript blocks for the user turn.
    Caps at 5 transcripts; each block capped at 800 chars to stay within budget.
    """
    blocks: list[str] = []
    for i, pt in enumerate(points[:5], start=1):
        p = pt.payload or {}
        title = p.get("title", "Unknown Lecture")
        start = p.get("start_sec", 0)
        text = (p.get("text", ""))[:800]
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


def _point_to_citation(pt: ScoredPoint) -> Citation:
    """Convert a Qdrant ScoredPoint to a Citation model."""
    p = pt.payload or {}
    text = p.get("text", "")
    preview = text[:120] + ("…" if len(text) > 120 else "")
    # Cosine distance = 1 - cosine_similarity; qdrant .score is similarity
    distance = 1.0 - float(pt.score)
    return Citation(
        video_id=p.get("video_id", ""),
        title=p.get("title", ""),
        start_sec=float(p.get("start_sec", 0)),
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
    query_processed: str = "",
    t0: float | None = None,
) -> RAGResponse:
    """
    Full answer synthesis pipeline with guardrails.

    Args:
        query:            Raw user query (for display in response).
        points:           Top-K ScoredPoints from Qdrant (ordered by score desc).
        query_processed:  Preprocessed query string (after acronym expansion).
        t0:               Pipeline start time (for latency_ms).

    Returns:
        RAGResponse — always a valid response, even on refusal or error.

    Guardrail sequence:
        1. If points is empty → refusal (0 tokens).
        2. If best cosine distance > MAX_DISTANCE → refusal (0 tokens).
        3. Build context, call Gemini, parse badge, filter citations.
    """
    if t0 is None:
        t0 = time.perf_counter()

    def _elapsed_ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    # ── Guard 1: no results at all ─────────────────────────────────────────
    if not points:
        return RAGResponse(
            answer=REFUSAL_MSG,
            is_refused=True,
            retrieval_distance=1.0,
            tokens_used=0,
            query_processed=query_processed or query,
            latency_ms=_elapsed_ms(),
        )

    # ── Guard 2: distance cutoff ───────────────────────────────────────────
    best_score = float(points[0].score)   # cosine similarity (0-1)
    best_distance = 1.0 - best_score
    if best_distance > config.MAX_DISTANCE:
        return RAGResponse(
            answer=REFUSAL_MSG,
            is_refused=True,
            retrieval_distance=round(best_distance, 4),
            tokens_used=0,
            query_processed=query_processed or query,
            latency_ms=_elapsed_ms(),
        )

    # ── Build context ──────────────────────────────────────────────────────
    context = _build_context(points)
    user_turn = (
        f"Lecture Transcripts:\n\n{context}\n\n"
        f"Question: {query_processed or query}"
    )

    # ── Call Gemini ────────────────────────────────────────────────────────
    client = _get_gemini_client()

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
        ),
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

    citations = [_point_to_citation(points[i]) for i in cited_indices]

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
