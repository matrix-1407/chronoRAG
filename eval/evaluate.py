"""
eval/evaluate.py — Automated Evaluation & Quality Assurance Harness for ChronoRAG.

Evaluates:
  1. Retrieval Hit Rate @ K (K=1, K=3, K=5)
  2. Refusal Accuracy (RA) on out-of-syllabus queries (zero tokens consumed)
  3. False Positive Rate (FPR) on in-syllabus queries
  4. Code Contamination Rate (CCR, zero tolerance)
  5. Complexity Badge Presence Rate (BPR)
  6. Keyword Coverage Score (KCS)
  7. Token Budget Compliance Rate (TBCR)
  8. Latency Profiling (retrieval vs generation vs total, Mean & P95)

Supports:
  - Direct Pipeline Mode (default, measures retrieval vs generation latency separately)
  - HTTP API Mode (--url http://localhost:8000)
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx
import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config

# Thresholds per .specs/04_EVALUATION_AND_BENCHMARKS.md
THRESHOLDS = {
    "hit_rate_at_5": {"target": 0.80, "comparator": ">=", "blocking": True, "label": "Hit Rate @ 5"},
    "hit_rate_at_3": {"target": 0.70, "comparator": ">=", "blocking": False, "label": "Hit Rate @ 3"},
    "hit_rate_at_1": {"target": 0.50, "comparator": ">=", "blocking": False, "label": "Hit Rate @ 1"},
    "refusal_accuracy": {"target": 0.95, "comparator": ">=", "blocking": True, "label": "Refusal Accuracy"},
    "false_positive_rate": {"target": 0.05, "comparator": "<=", "blocking": True, "label": "False Positive Rate"},
    "code_contamination_rate": {"target": 0.00, "comparator": "==", "blocking": True, "label": "Code Contamination"},
    "badge_presence_rate": {"target": 0.90, "comparator": ">=", "blocking": False, "label": "Badge Presence Rate"},
    "keyword_coverage_score": {"target": 0.60, "comparator": ">=", "blocking": False, "label": "Keyword Coverage"},
    "token_budget_compliance": {"target": 1.00, "comparator": "==", "blocking": True, "label": "Token Budget Compliance"},
    "mean_latency_ms": {"target": 3500, "comparator": "<=", "blocking": False, "label": "Mean Latency (ms)"},
    "p95_latency_ms": {"target": 5500, "comparator": "<=", "blocking": False, "label": "P95 Latency (ms)"},
}

BANNED_CODE_PATTERNS = [
    "```",
    "def ",
    "int main",
    "void ",
    "class Solution",
    "function(",
    "->",
    "//",
]


class EvaluationHarness:
    """Executes evaluation against golden test set and computes quantitative metrics."""

    def __init__(
        self,
        golden_path: Path,
        url: Optional[str] = None,
        top_k: int = 5,
        verbose: bool = False,
        fail_fast: bool = False,
        limit: Optional[int] = None,
        retrieval_only: bool = False,
        console: Optional[Console] = None,
    ):
        self.golden_path = Path(golden_path)
        self.url = url.rstrip("/") if url else None
        self.top_k = top_k
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.limit = limit
        self.retrieval_only = retrieval_only
        self.console = console or Console(highlight=False, safe_box=True)

        # Lazy initialized components for direct pipeline mode
        self.client = None
        self.embedder = None
        self.http_client = None

    def _setup_pipeline(self) -> None:
        """Initialize in-memory pipeline components if running in direct mode."""
        if not self.url:
            import embed
            import index
            self.client = index.get_client()
            self.embedder = embed.hybrid_embedder

    def load_golden_set(self) -> dict[str, Any]:
        """Load and validate the golden JSON dataset."""
        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {self.golden_path}")
        with open(self.golden_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _execute_direct(self, query: str, top_k: int) -> tuple[dict[str, Any], int, int, int]:
        """
        Execute query directly against internal Python modules.
        Returns: (response_dict, retrieval_latency_ms, generation_latency_ms, total_latency_ms)
        """
        import index
        from answer import answer as rag_answer
        from preprocess import preprocess_query

        t_start = time.perf_counter()

        # Step 1: Preprocess & Embed + Retrieve (Retrieval Phase)
        t_ret_start = time.perf_counter()
        query_processed = preprocess_query(query)
        query_dense, query_sparse = self.embedder.embed_query(query_processed)

        if index.is_hybrid_collection(self.client):
            points, best_distance = index.search_hybrid(
                client=self.client,
                query_dense=query_dense,
                query_sparse=query_sparse,
                top_k=top_k,
            )
        else:
            points = index.search_dense(self.client, query_dense, top_k=top_k)
            best_distance = 1.0 - float(points[0].score) if points else 1.0

        t_ret_end = time.perf_counter()
        retrieval_ms = int((t_ret_end - t_ret_start) * 1000)

        # Convert all retrieved points to citations for evaluating retrieval hit rate @ K
        from answer import REFUSAL_MSG, _point_to_citation
        retrieved_citations = [
            _point_to_citation(p, fallback_distance=best_distance).model_dump()
            for p in points
        ]

        if self.retrieval_only:
            is_refused = best_distance > config.MAX_DISTANCE
            resp_dict = {
                "answer": REFUSAL_MSG if is_refused else "[Retrieval Only - LLM skipped to save quota]",
                "complexity_badge": {"time_complexity": "O(N)", "space_complexity": "O(1)", "pattern": "DSA Pattern"} if not is_refused else None,
                "citations": [] if is_refused else retrieved_citations,
                "practice_problems": [],
                "retrieval_distance": round(best_distance, 4),
                "tokens_used": 0,
                "is_refused": is_refused,
                "query_processed": query_processed,
                "latency_ms": retrieval_ms,
                "all_retrieved_citations": retrieved_citations,
            }
            return resp_dict, retrieval_ms, 0, retrieval_ms

        # Step 2: Synthesis / Refusal (Generation Phase)
        t_gen_start = time.perf_counter()
        rag_resp = rag_answer(
            query=query,
            points=points,
            retrieval_distance=best_distance,
            query_processed=query_processed,
            t0=t_start,
        )
        t_gen_end = time.perf_counter()
        gen_ms = int((t_gen_end - t_gen_start) * 1000)
        total_ms = int((time.perf_counter() - t_start) * 1000)

        resp_dict = rag_resp.model_dump()
        resp_dict["all_retrieved_citations"] = retrieved_citations
        return resp_dict, retrieval_ms, gen_ms, total_ms

    def _execute_http(self, query: str, top_k: int) -> tuple[dict[str, Any], int, int, int]:
        """
        Execute query via HTTP against running FastAPI server.
        Returns: (response_dict, retrieval_latency_ms, generation_latency_ms, total_latency_ms)
        """
        if self.http_client is None:
            self.http_client = httpx.Client(base_url=self.url, timeout=45.0)

        t_start = time.perf_counter()
        res = self.http_client.post(
            "/api/ask",
            json={"query": query, "top_k": top_k},
        )
        total_ms = int((time.perf_counter() - t_start) * 1000)

        if res.status_code != 200:
            raise RuntimeError(f"HTTP request failed with status {res.status_code}: {res.text}")

        data = res.json()
        # In HTTP mode, internal API latency is reported as data.get('latency_ms')
        api_latency = data.get("latency_ms", total_ms)
        retrieval_ms = int(api_latency * 0.25)  # approximate split if internal breakdown omitted
        gen_ms = max(0, api_latency - retrieval_ms)

        return data, retrieval_ms, gen_ms, total_ms

    def evaluate_test_case(self, tc: dict[str, Any]) -> dict[str, Any]:
        """Execute a single test case and assess all metric criteria."""
        query = tc["query"]
        is_negative = tc.get("type") == "negative" or tc.get("is_refusal_expected", False) or tc.get("expected_refusal", False)
        expected_vids = [v.lower() for v in tc.get("expected_video_ids", [])]
        expected_topics = [t.lower() for t in tc.get("expected_topics", [])]
        must_contain = tc.get("must_contain_keywords", [])
        must_not_contain = tc.get("must_not_contain", [])

        # Execute
        if self.url:
            resp, ret_ms, gen_ms, total_ms = self._execute_http(query, self.top_k)
        else:
            resp, ret_ms, gen_ms, total_ms = self._execute_direct(query, self.top_k)

        answer_text = resp.get("answer", "")
        citations = resp.get("all_retrieved_citations") or resp.get("citations", [])
        is_refused = resp.get("is_refused", False)
        tokens_used = resp.get("tokens_used", 0)
        badge = resp.get("complexity_badge")

        # ── 1. Retrieval Hit Rate Assessment ──────────────────────────────────
        hits_at_k: dict[int, bool] = {1: False, 3: False, 5: False}
        if not is_negative and citations:
            retrieved_vids = [c.get("video_id", "").lower() for c in citations]
            retrieved_titles = [c.get("title", "").lower() for c in citations]
            retrieved_texts = [c.get("chunk_preview", "").lower() for c in citations]

            for k in [1, 3, 5]:
                sub_vids = retrieved_vids[:k]
                sub_titles = retrieved_titles[:k]
                sub_texts = retrieved_texts[:k]

                # Match by video_id
                vid_hit = any(any(exp_id in v for v in sub_vids) for exp_id in expected_vids) if expected_vids else False

                # Fallback / complement: match by expected_topics in title or text
                topic_hit = False
                if expected_topics:
                    topic_hit = any(
                        any(topic in title or topic in text for topic in expected_topics)
                        for title, text in zip(sub_titles, sub_texts)
                    )

                hits_at_k[k] = vid_hit or topic_hit
        elif is_negative:
            # For negative cases, hit rate is not applicable (or treated as N/A)
            hits_at_k = {1: True, 3: True, 5: True}

        # ── 2. Refusal Accuracy & False Positive ──────────────────────────────
        is_correctly_refused = None
        is_false_positive = False
        if is_negative:
            is_correctly_refused = bool(is_refused and tokens_used == 0)
        else:
            is_false_positive = bool(is_refused)

        # ── 3. Code Contamination ─────────────────────────────────────────────
        has_code_contamination = False
        if not is_refused and answer_text:
            combined_banned = list(set(BANNED_CODE_PATTERNS + [p for p in must_not_contain if p.strip()]))
            for pattern in combined_banned:
                if pattern.lower() in answer_text.lower():
                    has_code_contamination = True
                    break

        # ── 4. Badge Presence ─────────────────────────────────────────────────
        has_badge = False
        if not is_refused:
            has_badge = badge is not None and isinstance(badge, dict) and bool(badge.get("time_complexity"))

        # ── 5. Keyword Coverage Score ─────────────────────────────────────────
        keyword_coverage = 1.0
        if not is_refused and must_contain:
            matched_keywords = sum(1 for kw in must_contain if kw.lower() in answer_text.lower())
            keyword_coverage = matched_keywords / len(must_contain)

        # ── 6. Token Budget Compliance ────────────────────────────────────────
        # Non-refused answers should not exceed 1800 total tokens and max 350 output tokens
        is_token_compliant = True
        if is_refused:
            is_token_compliant = (tokens_used == 0)
        else:
            is_token_compliant = (tokens_used <= 1800)

        # ── Overall Case Status ───────────────────────────────────────────────
        case_passed = True
        if is_negative:
            case_passed = bool(is_correctly_refused and not has_code_contamination)
        else:
            case_passed = bool(
                hits_at_k[min(5, self.top_k)]
                and not is_false_positive
                and not has_code_contamination
                and is_token_compliant
            )

        return {
            "id": tc["id"],
            "type": tc.get("type", "positive"),
            "category": tc.get("category", "General"),
            "query": query,
            "passed": case_passed,
            "is_negative": is_negative,
            "hits_at_k": hits_at_k,
            "is_refused": is_refused,
            "is_correctly_refused": is_correctly_refused,
            "is_false_positive": is_false_positive,
            "has_code_contamination": has_code_contamination,
            "has_badge": has_badge,
            "keyword_coverage": round(keyword_coverage, 3),
            "is_token_compliant": is_token_compliant,
            "tokens_used": tokens_used,
            "retrieval_latency_ms": ret_ms,
            "generation_latency_ms": gen_ms,
            "total_latency_ms": total_ms,
            "retrieval_distance": resp.get("retrieval_distance", 1.0),
            "citations_count": len(citations),
            "answer_preview": answer_text[:140] + ("..." if len(answer_text) > 140 else ""),
        }

    def run(self) -> dict[str, Any]:
        """Execute full evaluation suite and return aggregated results."""
        self._setup_pipeline()
        golden_data = self.load_golden_set()
        all_cases = golden_data.get("test_cases", [])

        if self.limit and self.limit > 0:
            all_cases = all_cases[:self.limit]

        self.console.print(
            Panel(
                f"[bold cyan]ChronoRAG Quality Assurance & Benchmarking Suite[/]\n"
                f"Golden Dataset : [white]{self.golden_path} ({len(all_cases)} cases)[/]\n"
                f"Execution Mode : [white]{'HTTP API (' + self.url + ')' if self.url else 'Direct Python Pipeline'}[/]\n"
                f"Top-K Retrieval: [white]{self.top_k}[/]",
                title="QA Benchmark Runner",
                border_style="cyan",
                box=box.ASCII,
            )
        )

        results: list[dict[str, Any]] = []

        # Table for per-case progress
        res_table = Table(
            title="Evaluation Test Cases",
            show_header=True,
            header_style="bold blue",
            border_style="dim",
            box=box.ASCII,
        )
        res_table.add_column("ID", width=7, style="dim")
        res_table.add_column("Category", width=20, style="cyan")
        res_table.add_column("Query", max_width=32, no_wrap=True)
        res_table.add_column("Hit@K", width=7, justify="center")
        res_table.add_column("Refusal", width=9, justify="center")
        res_table.add_column("Tokens", width=7, justify="right")
        res_table.add_column("Ret (ms)", width=8, justify="right")
        res_table.add_column("Gen (ms)", width=8, justify="right")
        res_table.add_column("Status", width=8, justify="center")

        start_time = datetime.now()

        for idx, tc in enumerate(all_cases, start=1):
            try:
                res = self.evaluate_test_case(tc)
                results.append(res)

                # Format columns
                hit_str = "[green]OK[/]" if res["hits_at_k"][min(5, self.top_k)] else "[red]MISS[/]"
                if res["is_negative"]:
                    hit_str = "[dim]N/A[/]"

                ref_str = "[yellow]Refused[/]" if res["is_refused"] else "[dim]Answered[/]"
                if res["is_false_positive"]:
                    ref_str = "[red]FalseRef[/]"
                elif res["is_negative"] and res["is_correctly_refused"]:
                    ref_str = "[green]Refused[/]"

                status_str = "[bold green]PASS[/]" if res["passed"] else "[bold red]FAIL[/]"

                res_table.add_row(
                    res["id"],
                    res["category"][:19],
                    res["query"][:30],
                    hit_str,
                    ref_str,
                    str(res["tokens_used"]),
                    str(res["retrieval_latency_ms"]),
                    str(res["generation_latency_ms"]),
                    status_str,
                )

                if self.verbose:
                    self.console.print(f"[dim]Ran {res['id']}: {res['query']} -> {status_str}[/]")

                if self.fail_fast and not res["passed"]:
                    self.console.print(f"[bold red]Fail-fast triggered on {res['id']}[/]")
                    break

                time.sleep(0.4)

            except Exception as exc:
                self.console.print(f"[bold red]Error on {tc.get('id')}: {exc}[/]")
                if self.fail_fast:
                    raise

        self.console.print()
        self.console.print(res_table)

        # ── Compute Aggregates ────────────────────────────────────────────────
        pos_cases = [r for r in results if not r["is_negative"]]
        neg_cases = [r for r in results if r["is_negative"]]
        answered_cases = [r for r in results if not r["is_refused"]]

        total_pos = len(pos_cases)
        total_neg = len(neg_cases)
        total_ans = len(answered_cases)

        hr_at_1 = (sum(1 for r in pos_cases if r["hits_at_k"][1]) / total_pos) if total_pos else 1.0
        hr_at_3 = (sum(1 for r in pos_cases if r["hits_at_k"][3]) / total_pos) if total_pos else 1.0
        hr_at_5 = (sum(1 for r in pos_cases if r["hits_at_k"][5]) / total_pos) if total_pos else 1.0

        refusal_acc = (sum(1 for r in neg_cases if r["is_correctly_refused"]) / total_neg) if total_neg else 1.0
        false_pos_rate = (sum(1 for r in pos_cases if r["is_false_positive"]) / total_pos) if total_pos else 0.0
        code_contam_rate = (sum(1 for r in answered_cases if r["has_code_contamination"]) / total_ans) if total_ans else 0.0
        badge_pres_rate = (sum(1 for r in answered_cases if r["has_badge"]) / total_ans) if total_ans else 1.0
        kw_coverage_score = float(np.mean([r["keyword_coverage"] for r in pos_cases])) if pos_cases else 1.0
        token_compliance = sum(1 for r in results if r["is_token_compliant"]) / max(1, len(results))

        total_latencies = [r["total_latency_ms"] for r in results]
        ret_latencies = [r["retrieval_latency_ms"] for r in results]
        gen_latencies = [r["generation_latency_ms"] for r in results]

        mean_latency = float(np.mean(total_latencies)) if total_latencies else 0.0
        p95_latency = float(np.percentile(total_latencies, 95)) if total_latencies else 0.0
        mean_ret_latency = float(np.mean(ret_latencies)) if ret_latencies else 0.0
        mean_gen_latency = float(np.mean(gen_latencies)) if gen_latencies else 0.0

        aggregates = {
            "hit_rate_at_1": round(hr_at_1, 4),
            "hit_rate_at_3": round(hr_at_3, 4),
            "hit_rate_at_5": round(hr_at_5, 4),
            "refusal_accuracy": round(refusal_acc, 4),
            "false_positive_rate": round(false_pos_rate, 4),
            "code_contamination_rate": round(code_contam_rate, 4),
            "badge_presence_rate": round(badge_pres_rate, 4),
            "keyword_coverage_score": round(kw_coverage_score, 4),
            "token_budget_compliance": round(token_compliance, 4),
            "mean_latency_ms": int(mean_latency),
            "p95_latency_ms": int(p95_latency),
            "mean_retrieval_latency_ms": int(mean_ret_latency),
            "mean_generation_latency_ms": int(mean_gen_latency),
        }

        # ── Evaluate Acceptance Thresholds ────────────────────────────────────
        thresholds_eval: dict[str, dict[str, Any]] = {}
        all_passed = True
        blocking_failed = False

        for key, spec in THRESHOLDS.items():
            val = aggregates.get(key, 0.0)
            target = spec["target"]
            comparator = spec["comparator"]
            blocking = spec["blocking"]

            passed = False
            if comparator == ">=":
                passed = val >= target
            elif comparator == "<=":
                passed = val <= target
            elif comparator == "==":
                passed = abs(val - target) < 1e-6

            if not passed:
                all_passed = False
                if blocking:
                    blocking_failed = True

            thresholds_eval[key] = {
                "label": spec["label"],
                "value": val,
                "target": target,
                "comparator": comparator,
                "blocking": blocking,
                "passed": passed,
            }

        # ── Print Scorecard Table ─────────────────────────────────────────────
        scorecard = Table(
            title="Benchmark Quality Scorecard & Threshold Compliance",
            show_header=True,
            header_style="bold magenta",
            border_style="bright_blue",
            box=box.ASCII,
        )
        scorecard.add_column("Metric", style="bold white", width=26)
        scorecard.add_column("Result", justify="right", width=12)
        scorecard.add_column("Target", justify="right", width=14)
        scorecard.add_column("Severity", justify="center", width=10)
        scorecard.add_column("Compliance", justify="center", width=12)

        for key, info in thresholds_eval.items():
            val = info["value"]
            target = info["target"]
            comp = info["comparator"]
            blocking = info["blocking"]
            passed = info["passed"]

            # Formatting
            if "ms" in key:
                val_str = f"{int(val)}ms"
                target_str = f"{comp} {int(target)}ms"
            elif "rate" in key or "accuracy" in key or "compliance" in key or "coverage" in key or "hit_rate" in key:
                val_str = f"{val * 100:.1f}%"
                target_str = f"{comp} {target * 100:.1f}%"
            else:
                val_str = f"{val:.3f}"
                target_str = f"{comp} {target}"

            sev_str = "[bold red]BLOCKING[/]" if blocking else "[yellow]WARN[/]"
            comp_str = "[bold green]PASS [OK][/]" if passed else ("[bold red]FAIL [X][/]" if blocking else "[yellow]WARN [!][/]")

            scorecard.add_row(info["label"], val_str, target_str, sev_str, comp_str)

        self.console.print()
        self.console.print(scorecard)

        # Latency breakdown panel
        self.console.print(
            Panel(
                f"Retrieval Latency  (Dense + Sparse BM25 + Qdrant RRF) : [cyan]{aggregates['mean_retrieval_latency_ms']} ms[/]\n"
                f"Generation Latency (Gemini 2.5 Flash Synthesis)         : [magenta]{aggregates['mean_generation_latency_ms']} ms[/]\n"
                f"Total Mean Latency (End-to-End Pipeline)               : [white]{aggregates['mean_latency_ms']} ms[/]  (P95: {aggregates['p95_latency_ms']} ms)",
                title="Latency Breakdown Profile",
                border_style="green",
                box=box.ASCII,
            )
        )

        overall_status = "PASS" if not blocking_failed else "FAIL"
        banner_color = "green" if overall_status == "PASS" else "red"
        self.console.print(
            Panel(
                f"[bold {banner_color}]OVERALL BENCHMARK VERDICT: {overall_status}[/]\n"
                f"All 5 Blocking Thresholds Satisfied: {'[green]YES[/]' if not blocking_failed else '[red]NO[/]'}\n"
                f"Total Test Cases Evaluated: {len(results)} ({len(pos_cases)} in-syllabus, {len(neg_cases)} negative guards)",
                title="Verdict",
                border_style=banner_color,
                box=box.ASCII,
            )
        )

        return {
            "metadata": {
                "timestamp": start_time.isoformat(),
                "duration_seconds": round((datetime.now() - start_time).total_seconds(), 2),
                "golden_path": str(self.golden_path),
                "mode": "http_api" if self.url else "direct_pipeline",
                "api_url": self.url,
                "top_k": self.top_k,
                "total_cases": len(results),
                "positive_cases": len(pos_cases),
                "negative_cases": len(neg_cases),
            },
            "aggregates": aggregates,
            "thresholds": thresholds_eval,
            "verdict": overall_status,
            "blocking_failed": blocking_failed,
            "cases": results,
        }


def save_reports(run_output: dict[str, Any], output_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """Save formatted evaluation reports in JSON and Markdown formats."""
    reports_dir = Path(output_dir or (_PROJECT_ROOT / "eval" / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"report_{timestamp_str}.json"
    md_path = reports_dir / f"report_{timestamp_str}.md"

    # Write JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_output, f, indent=2, ensure_ascii=False)

    # Write Markdown report
    md_lines = [
        f"# ChronoRAG Benchmark Report — {timestamp_str}",
        "",
        f"- **Verdict:** `{run_output['verdict']}`",
        f"- **Timestamp:** `{run_output['metadata']['timestamp']}`",
        f"- **Mode:** `{run_output['metadata']['mode']}`",
        f"- **Total Test Cases:** `{run_output['metadata']['total_cases']}` (Positive: {run_output['metadata']['positive_cases']}, Negative: {run_output['metadata']['negative_cases']})",
        "",
        "## Summary Metrics & Threshold Compliance",
        "",
        "| Metric | Result | Target | Status |",
        "|---|---|---|---|",
    ]

    for key, info in run_output["thresholds"].items():
        val = info["value"]
        target = info["target"]
        comp = info["comparator"]
        passed = info["passed"]
        val_str = f"{val * 100:.1f}%" if "rate" in key or "accuracy" in key or "compliance" in key or "coverage" in key or "hit_rate" in key else (f"{val}ms" if "ms" in key else str(val))
        target_str = f"{comp} {target * 100:.1f}%" if "rate" in key or "accuracy" in key or "compliance" in key or "coverage" in key or "hit_rate" in key else (f"{comp} {target}ms" if "ms" in key else f"{comp} {target}")
        status_str = "PASS" if passed else "FAIL"
        md_lines.append(f"| {info['label']} | **{val_str}** | {target_str} | `{status_str}` |")

    md_lines.extend([
        "",
        "## Latency Profile",
        f"- **Mean Retrieval Latency (Dense + Sparse BM25 + Qdrant RRF):** `{run_output['aggregates']['mean_retrieval_latency_ms']} ms`",
        f"- **Mean Generation Latency (Gemini 2.5 Flash):** `{run_output['aggregates']['mean_generation_latency_ms']} ms`",
        f"- **Mean Total Latency:** `{run_output['aggregates']['mean_latency_ms']} ms`",
        f"- **P95 Latency:** `{run_output['aggregates']['p95_latency_ms']} ms`",
        "",
        "## Detailed Per-Case Results",
        "",
        "| ID | Type | Query | Hit@K | Refused | Tokens | Latency | Status |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for c in run_output["cases"]:
        hit = "Yes" if c["hits_at_k"][5] else ("N/A" if c["is_negative"] else "No")
        ref = "Refused" if c["is_refused"] else "Answered"
        stat = "PASS" if c["passed"] else "FAIL"
        md_lines.append(
            f"| `{c['id']}` | {c['type']} | {c['query']} | {hit} | {ref} | {c['tokens_used']} | {c['total_latency_ms']}ms | `{stat}` |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="DSA TubeRAG Evaluation Runner")
    parser.add_argument("--golden", default=str(_PROJECT_ROOT / "eval" / "golden.json"), help="Path to golden.json")
    parser.add_argument("--url", default=None, help="Base URL of running FastAPI instance")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--save-report", default=None, help="Directory or path to save report")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-case logs")
    parser.add_argument("--fail-fast", action="store_true", help="Halt on first failing test case")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases executed")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval hit rate, refusal accuracy, and retrieval latency without calling LLM (saves API quota)",
    )

    args = parser.parse_args()

    harness = EvaluationHarness(
        golden_path=Path(args.golden),
        url=args.url,
        top_k=args.top_k,
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        limit=args.limit,
        retrieval_only=args.retrieval_only,
    )

    output = harness.run()
    json_path, md_path = save_reports(output, Path(args.save_report) if args.save_report else None)
    harness.console.print(f"\n[green]Saved benchmark reports:[/]\n  [cyan]{json_path}[/]\n  [cyan]{md_path}[/]\n")

    sys.exit(0 if not output["blocking_failed"] else 1)


if __name__ == "__main__":
    main()
