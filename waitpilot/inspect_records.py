"""
inspect_records.py - readable side-by-side view of pilot_records.jsonl for manual review.

    python -m waitpilot.inspect_records > review.txt        # then open review.txt in VS Code
    python -m waitpilot.inspect_records --uid test/algebra/1234.json
"""
import argparse
import json
import sys
from collections import defaultdict

LINE = "=" * 100


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def show(text, width=None):
    return text if width is None else text[-width:]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="runs/calibration/pilot_records.jsonl")
    p.add_argument("--uid", help="only this problem")
    p.add_argument("--tail", type=int, default=800,
                   help="characters of the end of each output to show")
    p.add_argument("--context", type=int, default=400,
                   help="characters shown around the divergence point")
    args = p.parse_args()

    groups = defaultdict(dict)
    for r in load(args.path):
        if args.uid and r["unique_id"] != args.uid:
            continue
        groups[(r["unique_id"], r["seed"])][r["condition"]] = r

    for (uid, seed), conds in groups.items():
        any_r = next(iter(conds.values()))
        print(LINE)
        print(f"PROBLEM {uid}  (seed {seed}, {any_r['subject']}, level {any_r['level']})")
        print(f"GOLD ANSWER: {any_r['gold']}")
        print(LINE)

        for cond in ("baseline", "suppress"):
            r = conds.get(cond)
            if not r:
                continue
            print(f"\n--- {cond.upper()} ---")
            print(f"pred       : {r['pred']!r}")
            print(f"correct    : {r['correct']}    truncated: {r['truncated']}    "
                  f"finished_thinking: {r['finished_thinking']}    tokens: {r['n_tokens']}")
            print(f"markers    : {r['marker_counts']}")
            subs = {k: v for k, v in r['substitute_counts'].items() if v}
            print(f"substitutes: {subs}")
            out = r["output"]
            if "</think>" in out:
                final = out.split("</think>", 1)[1]
                print(f"\n[final answer section after </think>]\n{final.strip()}")
            else:
                print(f"\n[no </think> - last {args.tail} chars of output]\n"
                      f"...{show(out, args.tail)}")

        b, s = conds.get("baseline"), conds.get("suppress")
        if b and s:
            i = common_prefix_len(b["output"], s["output"])
            print(f"\n--- WHERE THE TWO RUNS DIVERGE (after {i} identical characters) ---")
            lo = max(0, i - args.context // 2)
            hi = i + args.context
            print(f"[shared text]   ...{b['output'][lo:i]}")
            print(f"[baseline then] {b['output'][i:hi]}...")
            print(f"[suppress then] {s['output'][i:hi]}...")
        print()


if __name__ == "__main__":
    main()
