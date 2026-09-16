"""
leetcode_mapper.py — Deterministic, token-free LeetCode & GFG problem auto-mapper.

Matches user queries, retrieved lecture titles, transcripts, and algorithmic patterns
to curated standard practice problems with direct URLs and difficulties.
Zero external API calls and zero LLM tokens consumed.
"""
from __future__ import annotations

import re
from typing import Any


# ── Explicit Problem ID to LeetCode Slug & Metadata ────────────────────────
LEETCODE_ID_MAP: dict[int, tuple[str, str, str]] = {
    # id: (title, slug, difficulty)
    1: ("Two Sum", "two-sum", "Easy"),
    2: ("Add Two Numbers", "add-two-numbers", "Medium"),
    3: ("Longest Substring Without Repeating Characters", "longest-substring-without-repeating-characters", "Medium"),
    11: ("Container With Most Water", "container-with-most-water", "Medium"),
    15: ("3Sum", "3sum", "Medium"),
    19: ("Remove Nth Node From End of List", "remove-nth-node-from-end-of-list", "Medium"),
    20: ("Valid Parentheses", "valid-parentheses", "Easy"),
    21: ("Merge Two Sorted Lists", "merge-two-sorted-lists", "Easy"),
    26: ("Remove Duplicates from Sorted Array", "remove-duplicates-from-sorted-array", "Easy"),
    27: ("Remove Element", "remove-element", "Easy"),
    33: ("Search in Rotated Sorted Array", "search-in-rotated-sorted-array", "Medium"),
    39: ("Combination Sum", "combination-sum", "Medium"),
    40: ("Combination Sum II", "combination-sum-ii", "Medium"),
    42: ("Trapping Rain Water", "trapping-rain-water", "Hard"),
    45: ("Jump Game II", "jump-game-ii", "Medium"),
    46: ("Permutations", "permutations", "Medium"),
    47: ("Permutations II", "permutations-ii", "Medium"),
    51: ("N-Queens", "n-queens", "Hard"),
    53: ("Kadane Maximum Subarray", "maximum-subarray", "Medium"),
    55: ("Jump Game", "jump-game", "Medium"),
    56: ("Merge Intervals", "merge-intervals", "Medium"),
    57: ("Insert Interval", "insert-interval", "Medium"),
    62: ("Unique Paths", "unique-paths", "Medium"),
    63: ("Unique Paths II", "unique-paths-ii", "Medium"),
    64: ("Minimum Path Sum", "minimum-path-sum", "Medium"),
    70: ("Climbing Stairs", "climbing-stairs", "Easy"),
    72: ("Edit Distance", "edit-distance", "Hard"),
    75: ("Sort Colors", "sort-colors", "Medium"),
    76: ("Minimum Window Substring", "minimum-window-substring", "Hard"),
    78: ("Subsets", "subsets", "Medium"),
    79: ("Word Search", "word-search", "Medium"),
    84: ("Largest Rectangle in Histogram", "largest-rectangle-in-histogram", "Hard"),
    90: ("Subsets II", "subsets-ii", "Medium"),
    98: ("Validate Binary Search Tree", "validate-binary-search-tree", "Medium"),
    100: ("Same Tree", "same-tree", "Easy"),
    101: ("Symmetric Tree", "symmetric-tree", "Easy"),
    102: ("Binary Tree Level Order Traversal", "binary-tree-level-order-traversal", "Medium"),
    104: ("Maximum Depth of Binary Tree", "maximum-depth-of-binary-tree", "Easy"),
    105: ("Construct Binary Tree from Preorder and Inorder Traversal", "construct-binary-tree-from-preorder-and-inorder-traversal", "Medium"),
    110: ("Balanced Binary Tree", "balanced-binary-tree", "Easy"),
    112: ("Path Sum", "path-sum", "Easy"),
    121: ("Best Time to Buy and Sell Stock", "best-time-to-buy-and-sell-stock", "Easy"),
    124: ("Binary Tree Maximum Path Sum", "binary-tree-maximum-path-sum", "Hard"),
    128: ("Longest Consecutive Sequence", "longest-consecutive-sequence", "Medium"),
    130: ("Surrounded Regions", "surrounded-regions", "Medium"),
    131: ("Palindrome Partitioning", "palindrome-partitioning", "Medium"),
    133: ("Clone Graph", "clone-graph", "Medium"),
    136: ("Single Number", "single-number", "Easy"),
    139: ("Word Break", "word-break", "Medium"),
    141: ("Linked List Cycle", "linked-list-cycle", "Easy"),
    142: ("Linked List Cycle II", "linked-list-cycle-ii", "Medium"),
    143: ("Reorder List", "reorder-list", "Medium"),
    148: ("Sort List", "sort-list", "Medium"),
    152: ("Maximum Product Subarray", "maximum-product-subarray", "Medium"),
    155: ("Min Stack", "min-stack", "Medium"),
    167: ("Two Sum II - Input Array Is Sorted", "two-sum-ii-input-array-is-sorted", "Medium"),
    169: ("Majority Element", "majority-element", "Easy"),
    190: ("Reverse Bits", "reverse-bits", "Easy"),
    191: ("Number of 1 Bits", "number-of-1-bits", "Easy"),
    198: ("House Robber", "house-robber", "Medium"),
    200: ("Number of Islands", "number-of-islands", "Medium"),
    206: ("Reverse Linked List", "reverse-linked-list", "Easy"),
    207: ("Course Schedule", "course-schedule", "Medium"),
    210: ("Course Schedule II", "course-schedule-ii", "Medium"),
    213: ("House Robber II", "house-robber-ii", "Medium"),
    215: ("Kth Largest Element in an Array", "kth-largest-element-in-an-array", "Medium"),
    225: ("Implement Stack using Queues", "implement-stack-using-queues", "Easy"),
    226: ("Invert Binary Tree", "invert-binary-tree", "Easy"),
    229: ("Majority Element II", "majority-element-ii", "Medium"),
    232: ("Implement Queue using Stacks", "implement-queue-using-stacks", "Easy"),
    234: ("Palindrome Linked List", "palindrome-linked-list", "Easy"),
    236: ("Lowest Common Ancestor of a Binary Tree", "lowest-common-ancestor-of-a-binary-tree", "Medium"),
    238: ("Product of Array Except Self", "product-of-array-except-self", "Medium"),
    239: ("Sliding Window Maximum", "sliding-window-maximum", "Hard"),
    261: ("Graph Valid Tree", "graph-valid-tree", "Medium"),
    268: ("Missing Number", "missing-number", "Easy"),
    283: ("Move Zeroes", "move-zeroes", "Easy"),
    287: ("Find the Duplicate Number", "find-the-duplicate-number", "Medium"),
    295: ("Find Median from Data Stream", "find-median-from-data-stream", "Hard"),
    300: ("Longest Increasing Subsequence", "longest-increasing-subsequence", "Medium"),
    303: ("Range Sum Query - Immutable", "range-sum-query-immutable", "Easy"),
    304: ("Range Sum Query 2D - Immutable", "range-sum-query-2d-immutable", "Medium"),
    322: ("Coin Change", "coin-change", "Medium"),
    323: ("Number of Connected Components in an Undirected Graph", "number-of-connected-components-in-an-undirected-graph", "Medium"),
    338: ("Counting Bits", "counting-bits", "Easy"),
    347: ("Top K Frequent Elements", "top-k-frequent-elements", "Medium"),
    416: ("Partition Equal Subset Sum", "partition-equal-subset-sum", "Medium"),
    417: ("Pacific Atlantic Water Flow", "pacific-atlantic-water-flow", "Medium"),
    435: ("Non-overlapping Intervals", "non-overlapping-intervals", "Medium"),
    455: ("Assign Cookies", "assign-cookies", "Easy"),
    485: ("Max Consecutive Ones", "max-consecutive-ones", "Easy"),
    496: ("Next Greater Element I", "next-greater-element-i", "Easy"),
    503: ("Next Greater Element II", "next-greater-element-ii", "Medium"),
    516: ("Longest Palindromic Subsequence", "longest-palindromic-subsequence", "Medium"),
    518: ("Coin Change II", "coin-change-ii", "Medium"),
    525: ("Contiguous Array", "contiguous-array", "Medium"),
    543: ("Diameter of Binary Tree", "diameter-of-binary-tree", "Easy"),
    560: ("Subarray Sum Equals K", "subarray-sum-equals-k", "Medium"),
    673: ("Number of Longest Increasing Subsequence", "number-of-longest-increasing-subsequence", "Medium"),
    684: ("Redundant Connection", "redundant-connection", "Medium"),
    700: ("Search in a Binary Search Tree", "search-in-a-binary-search-tree", "Easy"),
    704: ("Binary Search", "binary-search", "Easy"),
    739: ("Daily Temperatures", "daily-temperatures", "Medium"),
    743: ("Network Delay Time", "network-delay-time", "Medium"),
    746: ("Min Cost Climbing Stairs", "min-cost-climbing-stairs", "Easy"),
    778: ("Swim in Rising Water", "swim-in-rising-water", "Hard"),
    785: ("Is Graph Bipartite?", "is-graph-bipartite", "Medium"),
    787: ("Cheapest Flights Within K Stops", "cheapest-flights-within-k-stops", "Medium"),
    876: ("Middle of the Linked List", "middle-of-the-linked-list", "Easy"),
    912: ("Sort an Array", "sort-an-array", "Medium"),
    974: ("Subarray Sums Divisible by K", "subarray-sums-divisible-by-k", "Medium"),
    994: ("Rotting Oranges", "rotting-oranges", "Medium"),
    1143: ("Longest Common Subsequence", "longest-common-subsequence", "Medium"),
    1584: ("Min Cost to Connect All Points", "min-cost-to-connect-all-points", "Medium"),
}


# ── Curated Pattern & Topic to Practice Problems Dictionary ─────────────────
# Each entry maps keywords/patterns to 2-3 standard LeetCode & GFG problems.
CURATED_TOPIC_PRACTICE: list[dict[str, Any]] = [
    {
        "keywords": ["kadane", "maximum subarray", "max subarray", "contiguous subarray"],
        "problems": [
            {
                "title": "LeetCode #53: Maximum Subarray",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/maximum-subarray/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #152: Maximum Product Subarray",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/maximum-product-subarray/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: Kadane's Algorithm",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1",
                "difficulty": "Medium",
            },
        ],
    },
    {
        "keywords": ["lcs", "longest common subsequence", "common subsequence"],
        "problems": [
            {
                "title": "LeetCode #1143: Longest Common Subsequence",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/longest-common-subsequence/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #516: Longest Palindromic Subsequence",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/longest-palindromic-subsequence/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: Longest Common Subsequence",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/longest-common-subsequence-1587115620/1",
                "difficulty": "Medium",
            },
        ],
    },
    {
        "keywords": ["lis", "longest increasing subsequence", "increasing subsequence"],
        "problems": [
            {
                "title": "LeetCode #300: Longest Increasing Subsequence",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/longest-increasing-subsequence/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #673: Number of Longest Increasing Subsequence",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/number-of-longest-increasing-subsequence/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: Longest Increasing Subsequence",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/longest-increasing-subsequence-1587115620/1",
                "difficulty": "Medium",
            },
        ],
    },
    {
        "keywords": ["two pointer", "two-pointer", "two pointers", "2 pointer", "pair sum"],
        "problems": [
            {
                "title": "LeetCode #167: Two Sum II - Input Array Is Sorted",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #15: 3Sum",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/3sum/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #11: Container With Most Water",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/container-with-most-water/",
                "difficulty": "Medium",
            },
        ],
    },
    {
        "keywords": ["sliding window", "window maximum", "fixed window", "variable window"],
        "problems": [
            {
                "title": "LeetCode #3: Longest Substring Without Repeating Characters",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/longest-substring-without-repeating-characters/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #239: Sliding Window Maximum",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/sliding-window-maximum/",
                "difficulty": "Hard",
            },
            {
                "title": "GFG: Max Sum Subarray of size K",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/max-sum-subarray-of-size-k5313/1",
                "difficulty": "Easy",
            },
        ],
    },
    {
        "keywords": ["binary search", "search in rotated", "lower bound", "upper bound", "sorted array search"],
        "problems": [
            {
                "title": "LeetCode #704: Binary Search",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/binary-search/",
                "difficulty": "Easy",
            },
            {
                "title": "LeetCode #33: Search in Rotated Sorted Array",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/search-in-rotated-sorted-array/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: Binary Search",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/binary-search-1587115620/1",
                "difficulty": "Easy",
            },
        ],
    },
    {
        "keywords": ["prefix sum", "prefix-sum", "cumulative sum", "range sum query"],
        "problems": [
            {
                "title": "LeetCode #560: Subarray Sum Equals K",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/subarray-sum-equals-k/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #303: Range Sum Query - Immutable",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/range-sum-query-immutable/",
                "difficulty": "Easy",
            },
            {
                "title": "GFG: Subarray with given sum",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/subarray-with-given-sum-1587115621/1",
                "difficulty": "Medium",
            },
        ],
    },
    {
        "keywords": ["house robber", "house-robber", "adjacent elements dp"],
        "problems": [
            {
                "title": "LeetCode #198: House Robber",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/house-robber/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #213: House Robber II",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/house-robber-ii/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: Stickler Thief",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/stickler-theif-1587115621/1",
                "difficulty": "Medium",
            },
        ],
    },
    {
        "keywords": ["knapsack", "0/1 knapsack", "subset sum", "partition equal subset", "unbounded knapsack"],
        "problems": [
            {
                "title": "LeetCode #416: Partition Equal Subset Sum",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/partition-equal-subset-sum/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #322: Coin Change",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/coin-change/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: 0 - 1 Knapsack Problem",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/0-1-knapsack-problem0945/1",
                "difficulty": "Medium",
            },
        ],
    },
    {
        "keywords": ["dijkstra", "shortest path", "weighted graph shortest"],
        "problems": [
            {
                "title": "LeetCode #743: Network Delay Time",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/network-delay-time/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #787: Cheapest Flights Within K Stops",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/cheapest-flights-within-k-stops/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: Implementing Dijkstra Algorithm",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/implementing-dijkstra-set-1-adjacency-matrix/1",
                "difficulty": "Medium",
            },
        ],
    },
    {
        "keywords": ["cycle detection", "detect cycle", "cycle in graph", "cycle in linked list"],
        "problems": [
            {
                "title": "LeetCode #141: Linked List Cycle",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/linked-list-cycle/",
                "difficulty": "Easy",
            },
            {
                "title": "LeetCode #207: Course Schedule",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/course-schedule/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: Detect cycle in a directed graph",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/detect-cycle-in-a-directed-graph/1",
                "difficulty": "Medium",
            },
        ],
    },
    {
        "keywords": ["invert binary tree", "invert tree", "mirror tree"],
        "problems": [
            {
                "title": "LeetCode #226: Invert Binary Tree",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/invert-binary-tree/",
                "difficulty": "Easy",
            },
            {
                "title": "LeetCode #104: Maximum Depth of Binary Tree",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/maximum-depth-of-binary-tree/",
                "difficulty": "Easy",
            },
            {
                "title": "GFG: Mirror Tree",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/mirror-tree/1",
                "difficulty": "Easy",
            },
        ],
    },
    {
        "keywords": ["tree traversal", "bfs tree", "dfs tree", "level order", "binary tree"],
        "problems": [
            {
                "title": "LeetCode #102: Binary Tree Level Order Traversal",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/binary-tree-level-order-traversal/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #236: Lowest Common Ancestor of a Binary Tree",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: Height of Binary Tree",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/height-of-binary-tree/1",
                "difficulty": "Easy",
            },
        ],
    },
    {
        "keywords": ["bst", "binary search tree", "validate bst"],
        "problems": [
            {
                "title": "LeetCode #98: Validate Binary Search Tree",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/validate-binary-search-tree/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #700: Search in a Binary Search Tree",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/search-in-a-binary-search-tree/",
                "difficulty": "Easy",
            },
            {
                "title": "GFG: Check for BST",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/check-for-bst/1",
                "difficulty": "Easy",
            },
        ],
    },
    {
        "keywords": ["linked list", "reverse linked list", "middle of linked list", "palindrome linked list"],
        "problems": [
            {
                "title": "LeetCode #206: Reverse Linked List",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/reverse-linked-list/",
                "difficulty": "Easy",
            },
            {
                "title": "LeetCode #876: Middle of the Linked List",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/middle-of-the-linked-list/",
                "difficulty": "Easy",
            },
            {
                "title": "LeetCode #21: Merge Two Sorted Lists",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/merge-two-sorted-lists/",
                "difficulty": "Easy",
            },
        ],
    },
    {
        "keywords": ["monotonic stack", "next greater element", "daily temperatures", "histogram"],
        "problems": [
            {
                "title": "LeetCode #739: Daily Temperatures",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/daily-temperatures/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #496: Next Greater Element I",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/next-greater-element-i/",
                "difficulty": "Easy",
            },
            {
                "title": "LeetCode #84: Largest Rectangle in Histogram",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/largest-rectangle-in-histogram/",
                "difficulty": "Hard",
            },
        ],
    },
    {
        "keywords": ["backtracking", "recursion", "subsets", "combination sum", "n-queens"],
        "problems": [
            {
                "title": "LeetCode #78: Subsets",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/subsets/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #39: Combination Sum",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/combination-sum/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #51: N-Queens",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/n-queens/",
                "difficulty": "Hard",
            },
        ],
    },
    {
        "keywords": ["merge sort", "quick sort", "sort colors", "dutch national flag", "sorting"],
        "problems": [
            {
                "title": "LeetCode #912: Sort an Array",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/sort-an-array/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #75: Sort Colors",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/sort-colors/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: Merge Sort",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/merge-sort/1",
                "difficulty": "Medium",
            },
        ],
    },
    {
        "keywords": ["graph", "bfs", "dfs", "number of islands", "rotting oranges"],
        "problems": [
            {
                "title": "LeetCode #200: Number of Islands",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/number-of-islands/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #994: Rotting Oranges",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/rotting-oranges/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: BFS of graph",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/bfs-traversal-of-graph/1",
                "difficulty": "Easy",
            },
        ],
    },
    {
        "keywords": ["bit manipulation", "single number", "counting bits", "bitwise"],
        "problems": [
            {
                "title": "LeetCode #136: Single Number",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/single-number/",
                "difficulty": "Easy",
            },
            {
                "title": "LeetCode #338: Counting Bits",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/counting-bits/",
                "difficulty": "Easy",
            },
            {
                "title": "GFG: Set kth bit",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/problems/set-kth-bit3724/1",
                "difficulty": "Easy",
            },
        ],
    },
    {
        "keywords": ["dynamic programming", "memoization", "tabulation", "dp table", "1d dp", "2d dp"],
        "problems": [
            {
                "title": "LeetCode #70: Climbing Stairs",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/climbing-stairs/",
                "difficulty": "Easy",
            },
            {
                "title": "LeetCode #322: Coin Change",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/coin-change/",
                "difficulty": "Medium",
            },
            {
                "title": "LeetCode #1143: Longest Common Subsequence",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/longest-common-subsequence/",
                "difficulty": "Medium",
            },
        ],
    },
]


# ── LeetCode Number Detection Regexes ──────────────────────────────────────
_LC_NUMBER_PATTERNS = [
    re.compile(r"\bleetcode\s*#?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\blc\s*#?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bproblem\s*#?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"#(\d+)\b"),
]


def extract_leetcode_numbers(text: str) -> list[int]:
    """Extract distinct LeetCode problem numbers from a string."""
    numbers: list[int] = []
    seen: set[int] = set()
    for pat in _LC_NUMBER_PATTERNS:
        for match in pat.finditer(text):
            try:
                num = int(match.group(1))
                if num > 0 and num not in seen:
                    seen.add(num)
                    numbers.append(num)
            except (ValueError, IndexError):
                continue
    return numbers


def make_problem_from_id(num: int) -> dict[str, str]:
    """Construct a LeetCode problem link dictionary from problem ID."""
    if num in LEETCODE_ID_MAP:
        title, slug, diff = LEETCODE_ID_MAP[num]
        return {
            "title": f"LeetCode #{num}: {title}",
            "platform": "LeetCode",
            "url": f"https://leetcode.com/problems/{slug}/",
            "difficulty": diff,
        }
    # Fallback for arbitrary problem number
    return {
        "title": f"LeetCode #{num}",
        "platform": "LeetCode",
        "url": f"https://leetcode.com/problemset/all/?search={num}",
        "difficulty": "Medium",
    }


def get_practice_links(
    query: str,
    retrieved_titles: list[str] | None = None,
    retrieved_texts: list[str] | None = None,
    pattern: str | None = None,
    limit: int = 3,
) -> list[dict[str, str]]:
    """
    Topic-to-practice mapping engine.
    Returns 2-3 standard LeetCode and GeeksforGeeks practice problems matching the topic.

    Evaluation hierarchy:
    1. Explicit problem numbers mentioned in query or retrieved chunks.
    2. Pattern matched against ComplexityBadge pattern.
    3. Topic keywords matched against query, retrieved titles, and texts.

    Returns:
        List of dicts: [{"title": str, "platform": "LeetCode" | "GFG", "url": str, "difficulty": "Easy" | "Medium" | "Hard"}]
    """
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    def _add_problem(prob: dict[str, str]) -> None:
        url = prob.get("url", "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            results.append(prob)

    # 1. Search for explicit LeetCode problem numbers in query first
    query_nums = extract_leetcode_numbers(query)
    for num in query_nums:
        _add_problem(make_problem_from_id(num))

    # Also search for explicit numbers in retrieved titles
    if retrieved_titles:
        for t in retrieved_titles:
            for num in extract_leetcode_numbers(t):
                _add_problem(make_problem_from_id(num))

    # 2. Combine all search text for keyword/pattern matching
    search_corpus_parts = [query]
    if pattern:
        search_corpus_parts.append(pattern)
    if retrieved_titles:
        search_corpus_parts.extend(retrieved_titles)
    if retrieved_texts:
        # Include first 150 chars of top chunks to avoid searching excessive noise
        search_corpus_parts.extend([txt[:150] for txt in retrieved_texts[:3]])

    search_corpus = " ".join(search_corpus_parts).lower()

    # Match against curated patterns
    # Score each curated entry by keyword matches with heavy weight on query & pattern
    scored_entries: list[tuple[int, dict[str, Any]]] = []
    q_lower = query.lower()
    pat_lower = (pattern or "").lower()

    for entry in CURATED_TOPIC_PRACTICE:
        score = 0
        for kw in entry["keywords"]:
            words_count = len(kw.split())
            # Match directly in user's question (highest priority, quadratic with word length)
            if kw in q_lower:
                score += 25 * (words_count ** 2)
            # Match in detected algorithmic pattern
            elif pat_lower and kw in pat_lower:
                score += 10 * (words_count ** 2)
            # Match in retrieved lecture titles
            elif any(kw in t.lower() for t in (retrieved_titles or [])):
                score += 5 * words_count
            # Match in retrieved transcript text
            elif search_corpus and kw in search_corpus:
                score += 1 * words_count

        if score > 0:
            scored_entries.append((score, entry))

    # Sort entries by match strength descending
    scored_entries.sort(key=lambda x: x[0], reverse=True)

    for _, entry in scored_entries:
        for prob in entry["problems"]:
            _add_problem(prob)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    # If still empty, provide general DSA foundational practice
    if not results:
        default_fallback = [
            {
                "title": "LeetCode #1: Two Sum",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/two-sum/",
                "difficulty": "Easy",
            },
            {
                "title": "LeetCode #53: Maximum Subarray",
                "platform": "LeetCode",
                "url": "https://leetcode.com/problems/maximum-subarray/",
                "difficulty": "Medium",
            },
            {
                "title": "GFG: Top DSA Problem Set",
                "platform": "GFG",
                "url": "https://www.geeksforgeeks.org/explore?page=1&category=DSA",
                "difficulty": "Medium",
            },
        ]
        for p in default_fallback[:limit]:
            _add_problem(p)

    return results[:limit]
