"""smoke_test.py — Phase 1 module verification (no network calls)."""
import sys
from pathlib import Path

print("=== DSA TubeRAG Phase 1 — Smoke Test ===\n")

# 1. Config
import config
print(f"[1/5] config.py  OK  CHUNK_DURATION={config.CHUNK_DURATION}s  OVERLAP={config.CHUNK_OVERLAP}s  MAX_DISTANCE={config.MAX_DISTANCE}")

# 2. Models
from models import Chunk, Segment, TranscriptDoc, parse_badge, ComplexityBadge
cid = Chunk.make_id("testVid123", 0)
print(f"[2/5] models.py  OK  uuid5 example: {cid}")

# 3. Chunker on first 3 files
import chunk as ch
docs = []
for p in sorted(Path("transcripts").glob("*.json"))[:3]:
    d = ch.load_transcript(p)
    if d:
        docs.append(d)

all_chunks = ch.chunk_all(docs)
print(f"[3/5] chunk.py   OK  {len(docs)} docs -> {len(all_chunks)} chunks from 3 files")

if all_chunks:
    c = all_chunks[0]
    print(f"       id        : {c.id}")
    print(f"       video_id  : {c.video_id}")
    print(f"       title     : {c.title[:60]}")
    print(f"       window    : {c.start_sec}s - {c.end_sec}s  (seek @ {c.seek_sec}s)")
    print(f"       words     : {c.word_count}")
    print(f"       embed txt : {c.text_for_embed[:80]}...")

# 4. Badge parser
badge_text = "BFS level-by-level traversal hai. [Time: O(V+E) | Space: O(V) | Pattern: BFS]"
badge = parse_badge(badge_text)
assert badge is not None, "Badge parse failed!"
print(f"[4/5] models.py  OK  badge: time={badge.time_complexity} space={badge.space_complexity} pattern={badge.pattern}")

# 5. Full corpus stats (all 126 files)
print("[5/5] Loading all 126 transcripts for corpus stats ...")
all_docs = ch.load_all_transcripts()
all_ch = ch.chunk_all(all_docs)
avg = len(all_ch) / max(len(all_docs), 1)
total_words = sum(c.word_count for c in all_ch)

print(f"\n--- FULL CORPUS ---")
print(f"  JSON files loaded : {len(all_docs)}")
print(f"  Total chunks      : {len(all_ch):,}")
print(f"  Avg chunks/video  : {avg:.1f}")
print(f"  Total chunk words : {total_words:,}")
print(f"  Est. embed time   : ~{len(all_ch) // 32 + 1} batches @ batch_size=32")

# Verify idempotency: same chunk should produce same ID on re-run
if len(all_ch) >= 2:
    c1 = all_ch[0]
    reimported_id = Chunk.make_id(c1.video_id, c1.start_sec)
    assert reimported_id == c1.id, "Idempotency FAILED"
    print(f"\n  Idempotency check : PASSED (ID stable across calls)")

print("\n=== All checks PASSED ===")
