"""
cli.py — Typer + Rich CLI harness for DSA TubeRAG.

Commands:
  tuberag reindex   — chunk all transcripts and upsert to Qdrant
  tuberag stats     — show collection statistics
  tuberag search    — raw vector search (no LLM)
  tuberag ask       — full RAG pipeline (search + generate)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

import config

config.validate()   # Fail-fast on missing secrets

import chunk as chunker
import embed
import index
from answer import answer as rag_answer
from models import RAGResponse

app = typer.Typer(
    name="tuberag",
    help="DSA TubeRAG — Timestamp-accurate RAG for Padho with Pratyush",
    add_completion=False,
)
console = Console(highlight=False)


# ── reindex ────────────────────────────────────────────────────────────────

@app.command()
def reindex(
    transcripts_dir: Path = typer.Option(
        config.TRANSCRIPTS_DIR,
        "--dir", "-d",
        help="Directory containing transcript JSON files",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-index ALL videos, including already-indexed ones",
    ),
) -> None:
    """
    Chunk all transcripts and upsert embeddings to Qdrant Cloud.

    By default, videos already present in the collection are skipped
    (incremental sync). Use --force to wipe and re-index everything.
    """
    client = index.get_client()
    index.ensure_collection(client)

    # Determine which video_ids to skip
    if force:
        already_indexed: set[str] = set()
        console.print("[yellow]--force: re-indexing ALL videos[/]")
    else:
        console.print("[cyan]Checking already-indexed videos …[/]")
        already_indexed = index.get_indexed_video_ids(client)
        console.print(f"[cyan]Skipping {len(already_indexed)} already-indexed video(s)[/]")

    # Load & filter transcripts
    docs = chunker.load_all_transcripts(transcripts_dir)
    docs = [d for d in docs if d.video_id not in already_indexed]

    if not docs:
        console.print("[green]Nothing to index — all videos already up-to-date.[/]")
        raise typer.Exit()

    console.print(f"[bold]Indexing {len(docs)} video(s) …[/]")

    # Chunk
    chunks = chunker.chunk_all(docs)
    console.print(f"[cyan]Generated {len(chunks):,} chunks[/]")

    if not chunks:
        console.print("[yellow]No chunks produced — check transcript quality.[/]")
        raise typer.Exit()

    # Encode
    texts = [c.text_for_embed for c in chunks]
    vectors = embed.encode_chunks(texts)

    # Upsert
    total = index.upsert_chunks(client, chunks, vectors)
    console.print(
        Panel(
            f"[bold green][OK] Done![/]\n"
            f"Videos indexed : {len(docs)}\n"
            f"Chunks upserted: {total:,}\n"
            f"Collection     : {config.COLLECTION_NAME}",
            title="Reindex Complete",
        )
    )


# ── stats ──────────────────────────────────────────────────────────────────

@app.command()
def stats() -> None:
    """Show Qdrant collection statistics."""
    client = index.get_client()

    if not index.collection_exists(client):
        console.print(f"[red]Collection '{config.COLLECTION_NAME}' does not exist.[/]")
        console.print("Run [bold]tuberag reindex[/] first.")
        raise typer.Exit(1)

    info = index.get_collection_stats(client)
    indexed_ids = index.get_indexed_video_ids(client)

    table = Table(title=f"Collection: {config.COLLECTION_NAME}", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Total chunks", f"{info['total_chunks']:,}")
    table.add_row("Unique videos", str(len(indexed_ids)))
    table.add_row("Avg chunks/video",
                  f"{info['total_chunks'] / max(len(indexed_ids), 1):.1f}")
    table.add_row("Embed model", config.EMBED_MODEL)
    table.add_row("Embed dim", str(config.EMBED_DIM))
    table.add_row("Collection status", info["status"])
    console.print(table)


# ── search ─────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Query text to search for"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
) -> None:
    """Raw vector search — returns top-K chunks without LLM generation."""
    client = index.get_client()
    query_vec = embed.encode_query(query)
    results = index.search_dense(client, query_vec, top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit()

    table = Table(title=f"Search: '{query}'", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Score / Dist", width=14)
    table.add_column("Lecture", style="cyan")
    table.add_column("Timestamp", width=10)
    table.add_column("Preview", max_width=55, no_wrap=True)

    for i, pt in enumerate(results, 1):
        p = pt.payload or {}
        dist = round(1 - pt.score, 4)
        flag = " ✗" if dist > config.MAX_DISTANCE else ""
        table.add_row(
            str(i),
            f"{pt.score:.4f} / {dist:.4f}{flag}",
            p.get("title", "?")[:35],
            f"{int(p.get('start_sec', 0))}s",
            (p.get("text", ""))[:55],
        )

    console.print(table)

    best_dist = 1 - results[0].score
    if best_dist > config.MAX_DISTANCE:
        console.print(
            f"\n[red][WARN] Best distance {best_dist:.4f} > {config.MAX_DISTANCE} "
            f"(MAX_DISTANCE) - this query would be refused.[/]"
        )


# ── ask ────────────────────────────────────────────────────────────────────

@app.command()
def ask(
    query: str = typer.Argument(..., help="DSA question in Hindi/English/Hinglish"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
) -> None:
    """Full RAG pipeline: search + generate answer with Gemini."""
    client = index.get_client()
    t0 = time.perf_counter()

    # Retrieve
    query_vec = embed.encode_query(query)
    points = index.search_dense(client, query_vec, top_k=top_k)

    # Generate (includes refusal guard)
    resp: RAGResponse = rag_answer(query, points, t0=t0)

    # Display
    if resp.is_refused:
        console.print(Panel(
            f"[yellow]{resp.answer}[/]",
            title="[red][REFUSED] Out-of-syllabus[/]",
            subtitle=f"Distance: {resp.retrieval_distance} | Tokens: 0 | {resp.latency_ms}ms",
        ))
        return

    # Answer + badge
    badge_line = resp.complexity_badge.display() if resp.complexity_badge else ""
    content = resp.answer
    if badge_line:
        # Display badge separately from the text body for readability
        content = resp.answer.replace(badge_line, "").strip()

    console.print(Panel(content, title="Answer", border_style="green"))
    if badge_line:
        console.print(f"\n[bold magenta]{badge_line}[/]")

    # Citations
    if resp.citations:
        console.print()
        cite_table = Table(title="Sources", show_lines=False, box=None)
        cite_table.add_column("Ref", width=4, style="dim")
        cite_table.add_column("Lecture", style="cyan")
        cite_table.add_column("Timestamp", width=10)
        cite_table.add_column("Link", style="blue underline")
        for i, c in enumerate(resp.citations, 1):
            cite_table.add_row(
                f"[{i}]",
                c.title[:40],
                f"{int(c.start_sec)}s",
                c.youtube_url,
            )
        console.print(cite_table)

    console.print(
        f"\n[dim]Tokens: {resp.tokens_used} | "
        f"Distance: {resp.retrieval_distance} | "
        f"Latency: {resp.latency_ms}ms[/]"
    )


if __name__ == "__main__":
    app()
