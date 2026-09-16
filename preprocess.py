"""
preprocess.py — DSA query preprocessing and normalization.

Transforms raw user queries before embedding:
1. Standardizes LeetCode problem numbers (e.g., "lc 206", "leetcode #206" → "LeetCode 206").
2. Expands common DSA acronyms while retaining the original term
   (e.g., "LCS" → "LCS Longest Common Subsequence").
3. Normalizes whitespace and common formatting.
"""
from __future__ import annotations

import re


# ── DSA Acronym Map ────────────────────────────────────────────────────────
# Maps acronym to full canonical phrase.
DSA_ACRONYMS: dict[str, str] = {
    "LCS": "Longest Common Subsequence",
    "LIS": "Longest Increasing Subsequence",
    "BFS": "Breadth First Search",
    "DFS": "Depth First Search",
    "DP": "Dynamic Programming",
    "BST": "Binary Search Tree",
    "MST": "Minimum Spanning Tree",
    "KMP": "Knuth Morris Pratt",
    "AVL": "Adelson-Velsky Landis Tree",
    "GCD": "Greatest Common Divisor",
    "LCM": "Least Common Multiple",
    "GFG": "GeeksforGeeks",
    "LC": "LeetCode",
    "TLE": "Time Limit Exceeded",
    "MLE": "Memory Limit Exceeded",
    "SCC": "Strongly Connected Components",
    "DSU": "Disjoint Set Union",
    "BIT": "Binary Indexed Tree Fenwick Tree",
    "LRU": "Least Recently Used Cache",
    "LFU": "Least Frequently Used Cache",
    "2-SAT": "2 Satisfiability",
    "2SAT": "2 Satisfiability",
    "DAG": "Directed Acyclic Graph",
    "RMQ": "Range Minimum Query",
    "LCA": "Lowest Common Ancestor",
}


# ── Normalization Functions ────────────────────────────────────────────────

def normalize_leetcode(text: str) -> str:
    """
    Standardize LeetCode problem references and enrich with canonical titles.
    Examples:
        "lc 206"        → "LeetCode 206 Reverse Linked List"
        "leetcode #1"   → "LeetCode 1 Two Sum"
        "lc#15"         → "LeetCode 15 3Sum"
    """
    pattern = re.compile(r"\b(?:lc|leetcode)\s*#?\s*(\d+)\b", re.IGNORECASE)

    def _repl(m: re.Match) -> str:
        num_str = m.group(1)
        try:
            from leetcode_mapper import LEETCODE_ID_MAP
            num = int(num_str)
            if num in LEETCODE_ID_MAP:
                title = LEETCODE_ID_MAP[num][0]
                if title.lower() not in text.lower():
                    return f"LeetCode {num_str} {title}"
        except Exception:
            pass
        return f"LeetCode {num_str}"

    return pattern.sub(_repl, text)


def expand_acronyms(text: str) -> str:
    """
    Expand known DSA acronyms while retaining the original term.
    If the expansion is already present in the text, leaves it alone.

    Example:
        "how does LCS work?" → "how does LCS Longest Common Subsequence work?"
        "LCS Longest Common Subsequence" → unchanged (no duplicate expansion)
    """
    result = text
    lower_text = text.lower()

    for acronym, expansion in DSA_ACRONYMS.items():
        # Avoid expanding if the expansion is already in the query
        if expansion.lower() in lower_text:
            continue

        # Match whole word only (case-insensitive)
        pattern = re.compile(rf"\b{re.escape(acronym)}\b", re.IGNORECASE)
        match = pattern.search(result)
        if match:
            matched_term = match.group(0)
            replacement = f"{matched_term} {expansion}"
            result = pattern.sub(replacement, result, count=1)
            lower_text = result.lower()

    return result


# ── DSA Concept & Phrase Normalization ─────────────────────────────────────
# Maps common technical phrasing to canonical vocabulary used in lecture transcripts
# (e.g. bridging Hinglish questions to DP table initialization and base case terminology).
DSA_CONCEPT_EXPANSIONS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(?:table\s*init(?:ialization)?|initializ(?:e|ation)\s*(?:of\s*)?(?:the\s*)?table|dp\s*table|tabulation\s*table|table\s*(?:kaise|banate|banae|create|fill))\b",
            re.IGNORECASE,
        ),
        "tabulation dp table initialization base case row column 0",
    ),
    (
        re.compile(r"\b(?:memo(?:ization)?|memoize)\b", re.IGNORECASE),
        "memoization top down recursion dp array cache",
    ),
]


def expand_concepts(text: str) -> str:
    """
    Enrich query with canonical concept terms to improve dense and sparse (BM25)
    matching against lecture transcripts without altering original user intent.
    """
    result = text
    lower_text = text.lower()
    for pattern, expansion in DSA_CONCEPT_EXPANSIONS:
        if pattern.search(result):
            additions = [w for w in expansion.split() if w.lower() not in lower_text]
            if additions:
                result = f"{result} {' '.join(additions)}"
                lower_text = result.lower()
    return result


def preprocess_query(query: str) -> str:
    """
    Full preprocessing pipeline for a user search/RAG query.
    1. Strip & normalize excess spaces
    2. Normalize LeetCode references
    3. Expand DSA acronyms
    4. Expand canonical DSA concepts and technical vocabulary
    """
    if not query:
        return ""

    q = re.sub(r"\s+", " ", query.strip())
    q = normalize_leetcode(q)
    q = expand_acronyms(q)
    q = expand_concepts(q)
    return q.strip()

