"""
summarize.py - quick look at runs/calibration/pilot_records.jsonl.

    python -m waitpilot.summarize
    python -m waitpilot.summarize --show-leaks      # print the context of every leaked marker
"""
import argparse
import json
import sys
from collections import defaultdict

from waitpilot.common import MARKER_RE   # same pattern pilot.py and the cloud pipeline count with


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="runs/calibration/pilot_records.jsonl")
    p.add_argument("--show-leaks", action="store_true")
    args = p.parse_args()

    recs = load(args.path)
    by_cond = defaultdict(list)
    for r in recs:
        by_cond[r["condition"]].append(r)

    print(f"{len(recs)} records in {args.path}\n")
    # acc      = correct / all runs (a truncated run counts as wrong)
    # acc_fin  = correct / runs that finished (not truncated)
    print(f"{'condition':<10}{'n':>5}{'acc':>7}{'acc_fin':>9}{'trunc':>7}{'tokens':>9}"
          f"{'markers':>9}{'substit.':>10}{'tok/s':>8}")
    for cond, rs in sorted(by_cond.items()):
        fin = [r for r in rs if not r["truncated"]]
        acc_fin = f"{mean(r['correct'] for r in fin):.0%}" if fin else "-"
        print(f"{cond:<10}{len(rs):>5}"
              f"{mean(r['correct'] for r in rs):>7.0%}"
              f"{acc_fin:>9}"
              f"{mean(r['truncated'] for r in rs):>7.0%}"
              f"{mean(r['n_tokens'] for r in rs):>9.0f}"
              f"{mean(sum(r['marker_counts'].values()) for r in rs):>9.1f}"
              f"{mean(sum(r['substitute_counts'].values()) for r in rs):>10.1f}"
              f"{mean(r['tokens_per_s'] or 0 for r in rs):>8.1f}")

    # Substitute phrases, per condition
    print("\nsubstitute phrases (mean per trace):")
    keys = sorted({k for r in recs for k in r["substitute_counts"]})
    for cond, rs in sorted(by_cond.items()):
        parts = [f"{k}={mean(r['substitute_counts'].get(k, 0) for r in rs):.1f}" for k in keys]
        print(f"  {cond:<10}" + "  ".join(parts))

    # Paired table: same problem + seed in both conditions
    pairs = defaultdict(dict)
    for r in recs:
        pairs[(r["unique_id"], r["seed"])][r["condition"]] = r
    both = [v for v in pairs.values() if "baseline" in v and "suppress" in v]
    if both:
        bb = sum(v["baseline"]["correct"] and v["suppress"]["correct"] for v in both)
        bo = sum(v["baseline"]["correct"] and not v["suppress"]["correct"] for v in both)
        so = sum(not v["baseline"]["correct"] and v["suppress"]["correct"] for v in both)
        nn = len(both) - bb - bo - so
        print(f"\npaired ({len(both)} pairs):  both right={bb}  baseline only={bo}  "
              f"suppress only={so}  both wrong={nn}")
        both_finished = [v for v in both if not v["baseline"]["truncated"]
                         and not v["suppress"]["truncated"]]
        print(f"pairs where neither run was truncated: {len(both_finished)}")
        lost_to_trunc = sum(v["baseline"]["correct"] and v["suppress"]["truncated"]
                            for v in both)
        lost_to_trunc_b = sum(v["suppress"]["correct"] and v["baseline"]["truncated"]
                              for v in both)
        print(f"baseline right, suppress truncated: {lost_to_trunc}   "
              f"suppress right, baseline truncated: {lost_to_trunc_b}")

    # Leak check: markers appearing in the suppressed condition
    leaks = [r for r in by_cond.get("suppress", []) if sum(r["marker_counts"].values())]
    print(f"\nsuppress runs containing a marker in the thinking: {len(leaks)}"
          f" / {len(by_cond.get('suppress', []))}")
    if leaks and args.show_leaks:
        for r in leaks:
            thinking = r["output"].split("</think>", 1)[0]
            for m in MARKER_RE.finditer(thinking):
                a, b = max(0, m.start() - 60), m.end() + 60
                ctx = thinking[a:b].replace("\n", " ")
                print(f"  {r['unique_id']}: ...{ctx}...")


if __name__ == "__main__":
    main()
