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
    Standardize LeetCode problem references.
    Examples:
        "lc 206"        → "LeetCode 206"
        "leetcode #1"   → "LeetCode 1"
        "lc#15"         → "LeetCode 15"
    """
    pattern = re.compile(r"\b(?:lc|leetcode)\s*#?\s*(\d+)\b", re.IGNORECASE)
    return pattern.sub(r"LeetCode \1", text)


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


def preprocess_query(query: str) -> str:
    """
    Full preprocessing pipeline for a user search/RAG query.
    1. Strip & normalize excess spaces
    2. Normalize LeetCode references
    3. Expand DSA acronyms
    """
    if not query:
        return ""

    q = re.sub(r"\s+", " ", query.strip())
    q = normalize_leetcode(q)
    q = expand_acronyms(q)
    return q.strip()
