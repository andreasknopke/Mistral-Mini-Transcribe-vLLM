import json, sys
from scripts.ssh_helper import get_client

c = get_client()
stdin, stdout, stderr = c.exec_command("curl -s http://localhost:8000/debug/last_response", timeout=15)
data = json.loads(stdout.read().decode())
c.close()

r = data["response"]
segs = r.get("segments", [])
print(f"Segments: {len(segs)}, duration: {r.get('duration')}")

# Flatten all words
all_words = []
for s in segs:
    all_words.extend(s.get("words", []))

print(f"Total words: {len(all_words)}")

# Find where chunk boundary is (jump from <180 to >=180)
print("\n--- Words around 180s boundary ---")
for i, w in enumerate(all_words):
    if 170 < w["start"] < 195 or (i > 0 and all_words[i-1]["end"] < 180 and w["start"] >= 180):
        print(f"  Word {i:3d}: {w['word']:35s} {w['start']:7.2f}-{w['end']:7.2f} "
              f"dur={w['end']-w['start']:.3f}")

# Find collapse: words with near-zero duration in sequence
print("\n--- Collapsed regions (5+ zero-dur words in a row) ---")
run_start = None
run_count = 0
for i, w in enumerate(all_words):
    dur = w["end"] - w["start"]
    if dur < 0.02:
        if run_start is None:
            run_start = i
        run_count += 1
    else:
        if run_count >= 5:
            print(f"  Words {run_start}-{run_start+run_count-1}: "
                  f"ts={all_words[run_start]['start']:.2f}-{all_words[run_start+run_count-1]['end']:.2f} "
                  f"({run_count} words collapsed)")
            # Show context
            for j in range(max(0, run_start-2), min(len(all_words), run_start+run_count+2)):
                ww = all_words[j]
                flag = " ***" if ww["end"] - ww["start"] < 0.02 else ""
                print(f"    {j:3d}: {ww['word']:35s} {ww['start']:7.2f}-{ww['end']:7.2f}{flag}")
        run_start = None
        run_count = 0

if run_count >= 5:
    print(f"  Words {run_start}-{run_start+run_count-1}: ({run_count} collapsed at end)")

# Show segments summary
print("\n--- Segments ---")
for i, s in enumerate(segs):
    ws = s.get("words", [])
    dur = s["end"] - s["start"]
    collapsed = sum(1 for w in ws if w["end"] - w["start"] < 0.02)
    flag = " ***" if collapsed > 3 else ""
    print(f"  Seg {i:2d}: {s['start']:7.2f}-{s['end']:7.2f} ({dur:5.1f}s) "
          f"w={len(ws):2d} coll={collapsed}{flag} | {s['text'][:60]}")

