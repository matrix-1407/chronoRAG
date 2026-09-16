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
from preprocess import preprocess_query

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
        help="Re-index ALL videos and recreate collection with hybrid named-vectors schema",
    ),
) -> None:
    """
    Chunk all transcripts and upsert dense + sparse embeddings to Qdrant Cloud.

    By default, videos already present in the collection are skipped
    (incremental sync). Use --force to wipe and re-index with the Phase 2
    hybrid vector schema.
    """
    client = index.get_client()

    # Recreate collection if forced or if existing schema is non-hybrid
    needs_recreate = force or (index.collection_exists(client) and not index.is_hybrid_collection(client))
    index.ensure_collection(client, force_recreate=needs_recreate)

    # Determine which video_ids to skip
    if force or needs_recreate:
        already_indexed: set[str] = set()
        console.print("[yellow]Re-indexing ALL videos into hybrid collection schema[/]")
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

    # Hybrid encode (dense bge-m3 + sparse BM25)
    texts = [c.text_for_embed for c in chunks]
    dense_vectors, sparse_vectors = embed.hybrid_embedder.embed_chunks(texts)

    # Upsert with named vectors
    total = index.upsert_chunks(client, chunks, dense_vectors, sparse_vectors)
    console.print(
        Panel(
            f"[bold green][OK] Done![/]\n"
            f"Videos indexed : {len(docs)}\n"
            f"Chunks upserted: {total:,}\n"
            f"Collection     : {config.COLLECTION_NAME}\n"
            f"Schema         : Hybrid (Dense 1024 Cosine + Sparse BM25)",
            title="Hybrid Reindex Complete",
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


# ── search / find ──────────────────────────────────────────────────────────

@app.command(name="find")
@app.command(name="search")
def find(
    query: str = typer.Argument(..., help="Query text to search for"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
) -> None:
    """
    Search lecture chunks — displays RRF, dense, and sparse matching scores.
    """
    client = index.get_client()

    # Preprocess
    query_processed = preprocess_query(query)
    if query_processed.lower() != query.lower():
        console.print(f"[dim]Expanded query:[/] [cyan]{query_processed}[/]")

    dense_vec, sparse_vec = embed.hybrid_embedder.embed_query(query_processed)

    if index.is_hybrid_collection(client):
        # Retrieve dense and sparse independently to extract individual scores
        dense_results = index.search_dense(client, dense_vec, top_k=top_k * 2)
        sparse_results = index.search_sparse(client, sparse_vec, top_k=top_k * 2)
        hybrid_results, best_dist = index.search_hybrid(
            client=client,
            query_dense=dense_vec,
            query_sparse=sparse_vec,
            top_k=top_k,
        )

        if not hybrid_results:
            console.print("[yellow]No results found.[/]")
            raise typer.Exit()

        dense_scores = {pt.id: float(pt.score) for pt in dense_results}
        sparse_scores = {pt.id: float(pt.score) for pt in sparse_results}

        table = Table(title=f"Hybrid Search (RRF): '{query}'", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("RRF Score", width=10, style="magenta")
        table.add_column("Dense (Cosine)", width=14, style="green")
        table.add_column("Sparse (BM25)", width=13, style="yellow")
        table.add_column("Lecture", style="cyan")
        table.add_column("Timestamp", width=10)
        table.add_column("Preview", max_width=50, no_wrap=True)

        for i, pt in enumerate(hybrid_results, 1):
            p = pt.payload or {}
            d_score = dense_scores.get(pt.id)
            d_str = f"{d_score:.3f} (d={1-d_score:.3f})" if d_score is not None else "—"
            s_score = sparse_scores.get(pt.id)
            s_str = f"{s_score:.2f}" if s_score is not None else "—"

            table.add_row(
                str(i),
                f"{pt.score:.4f}",
                d_str,
                s_str,
                p.get("title", "?")[:32],
                f"{int(p.get('start_sec', 0))}s",
                (p.get("text", ""))[:50],
            )
        console.print(table)

        if best_dist > config.MAX_DISTANCE:
            console.print(
                f"\n[red][WARN] Best dense distance {best_dist:.4f} > {config.MAX_DISTANCE} "
                f"(MAX_DISTANCE) — this query would trigger refusal guard.[/]"
            )
    else:
        # Fallback to dense search if collection not yet migrated
        results = index.search_dense(client, dense_vec, top_k=top_k)
        if not results:
            console.print("[yellow]No results found.[/]")
            raise typer.Exit()

        table = Table(title=f"Dense Search: '{query}'", show_lines=True)
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
                f"(MAX_DISTANCE) — this query would trigger refusal guard.[/]"
            )


# ── ask ────────────────────────────────────────────────────────────────────

@app.command()
def ask(
    query: str = typer.Argument(..., help="DSA question in Hindi/English/Hinglish"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
) -> None:
    """Full RAG pipeline: preprocess + hybrid search + generate answer with Gemini."""
    client = index.get_client()
    t0 = time.perf_counter()
    timing: dict[str, float] = {}

    # Preprocess
    query_processed = preprocess_query(query)
    if query_processed.lower() != query.lower():
        console.print(f"[dim]Expanded query:[/] [cyan]{query_processed}[/]")

    # Hybrid Embed & Retrieve (with high-resolution latency measurement)
    t_embed_start = time.perf_counter()
    dense_vec, sparse_vec = embed.hybrid_embedder.embed_query(query_processed)
    timing["embed_ms"] = round((time.perf_counter() - t_embed_start) * 1000, 2)

    t_ret_start = time.perf_counter()
    if index.is_hybrid_collection(client):
        points, best_distance = index.search_hybrid(
            client=client,
            query_dense=dense_vec,
            query_sparse=sparse_vec,
            top_k=top_k,
        )
    else:
        points = index.search_dense(client, dense_vec, top_k=top_k)
        best_distance = 1.0 - float(points[0].score) if points else 1.0
    timing["retrieval_ms"] = round((time.perf_counter() - t_ret_start) * 1000, 2)

    # Generate (includes refusal guard, timing generation_ms and total_ms)
    resp: RAGResponse = rag_answer(
        query=query,
        points=points,
        retrieval_distance=best_distance,
        query_processed=query_processed,
        t0=t0,
        timing=timing,
    )

    # Display
    if resp.is_refused:
        timing_sub = ""
        if resp.timing:
            t = resp.timing
            timing_sub = (
                f" | Embed: {t.get('embed_ms', 0):.1f}ms | "
                f"Ret: {t.get('retrieval_ms', 0):.1f}ms | Gen: 0ms | "
                f"Total: {t.get('total_ms', 0):.1f}ms"
            )
        console.print(Panel(
            f"[yellow]{resp.answer}[/]",
            title="[red][REFUSED] Out-of-syllabus[/]",
            subtitle=f"Distance: {resp.retrieval_distance} | Tokens: 0{timing_sub}",
        ))
        return

    # Answer + badge
    badge_line = resp.complexity_badge.display() if resp.complexity_badge else ""
    content = resp.answer
    if badge_line:
        # Display badge separately from the text body for readability
        content = resp.answer.replace(badge_line, "").strip()

    answer_title = f"Answer ({resp.model_used})" if resp.model_used else "Answer"
    console.print(Panel(content, title=answer_title, border_style="green"))
    if badge_line:
        console.print(f"\n[bold magenta]{badge_line}[/]")

    # Millisecond latency breakdown directly below badge
    if resp.timing:
        t = resp.timing
        console.print(
            f"[dim]Timing breakdown:[/] "
            f"Embed: [bold cyan]{t.get('embed_ms', 0):.1f}ms[/] | "
            f"Retrieval: [bold cyan]{t.get('retrieval_ms', 0):.1f}ms[/] | "
            f"Generation: [bold cyan]{t.get('generation_ms', 0):.1f}ms[/] | "
            f"Total: [bold green]{t.get('total_ms', 0):.1f}ms[/]"
        )

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

    # Practice Problems (Phase 4)
    if resp.practice_problems:
        console.print()
        prob_table = Table(title="Practice Problems", show_lines=False, box=None)
        prob_table.add_column("Platform", width=10, style="bold cyan")
        prob_table.add_column("Problem", style="white")
        prob_table.add_column("Difficulty", width=10)
        prob_table.add_column("URL", style="blue underline")
        for p in resp.practice_problems:
            diff_style = "green" if p.difficulty == "Easy" else ("yellow" if p.difficulty == "Medium" else "red")
            prob_table.add_row(
                p.platform,
                p.title,
                f"[{diff_style}]{p.difficulty}[/]",
                p.url,
            )
        console.print(prob_table)

    console.print(
        f"\n[dim]Tokens: {resp.tokens_used} | "
        f"Distance: {resp.retrieval_distance} | "
        f"Latency: {resp.latency_ms}ms[/]"
    )


# ── benchmark (Phase 5) ────────────────────────────────────────────────────

@app.command()
def benchmark(
    golden: Path = typer.Option(
        Path("eval/golden.json"),
        "--golden", "-g",
        help="Path to golden evaluation dataset JSON",
    ),
    save_report: Optional[Path] = typer.Option(
        None,
        "--save-report", "-s",
        help="Directory or path to save output JSON/Markdown report (default: eval/reports/)",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
    url: Optional[str] = typer.Option(
        None, "--url", "-u", help="Base URL of running FastAPI instance (default: direct pipeline)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-case details"),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Halt on first failing test case"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit number of test cases to run"),
    retrieval_only: bool = typer.Option(
        False,
        "--retrieval-only", "-r",
        help="Evaluate retrieval hit rate, refusal accuracy, and retrieval latency without calling LLM (saves API quota)",
    ),
) -> None:
    """
    Run automated QA and benchmarking suite against the golden dataset.
    Computes Hit Rate @ K, Refusal Accuracy, Code Contamination, Latency Profiling,
    and Token Compliance, saving formatted reports to eval/reports/.
    """
    from eval.evaluate import EvaluationHarness, save_reports

    harness = EvaluationHarness(
        golden_path=golden,
        url=url,
        top_k=top_k,
        verbose=verbose,
        fail_fast=fail_fast,
        limit=limit,
        retrieval_only=retrieval_only,
        console=console,
    )

    output = harness.run()
    json_path, md_path = save_reports(output, save_report)
    console.print(f"\n[green]Saved benchmark reports:[/]\n  [cyan]{json_path}[/]\n  [cyan]{md_path}[/]\n")

    if output.get("blocking_failed"):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

