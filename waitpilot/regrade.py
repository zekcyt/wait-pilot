"""
regrade.py - offline regrading of any records file that has "output" and "gold"
(pilot.py records, cloud_generate.py baselines and branches).

Uses grading.py: every box in the final section, sets compared order-insensitively,
bracketed answers keep their order, math-verify equivalence (pip install math-verify).

    python -m waitpilot.regrade runs/calibration/calib_l123_para.jsonl
    python -m waitpilot.regrade runs/calibration/calib_l123_para.jsonl --out runs/calibration/calib_l123_para.regraded.jsonl
    python -m waitpilot.regrade runs/main/branches.jsonl --in-place

The original grade is kept as correct_old / pred_old, so nothing is lost. Prints every
record whose grade changed, for a manual check.
"""
import argparse
import json
import os
import sys

from waitpilot.grading import GRADER_VERSION, HAVE_MATH_VERIFY, grade_output


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--out", help="default: <path>.regraded.jsonl")
    p.add_argument("--in-place", action="store_true")
    args = p.parse_args()

    if not HAVE_MATH_VERIFY:
        print("WARNING: math-verify is not installed; grading with the strict legacy "
              "matcher only. pip install math-verify\n")

    with open(args.path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]

    flips = []
    n_graded = 0
    for i, r in enumerate(recs, 1):
        if "output" not in r or "gold" not in r:
            continue
        n_graded += 1
        g = grade_output(r["output"], r["gold"])
        old_c, old_p = r.get("correct_old", r.get("correct")), r.get("pred_old", r.get("pred"))
        r["correct_old"], r["pred_old"] = old_c, old_p
        r.update(g)
        if bool(old_c) != bool(g["correct"]):
            flips.append((i, r, old_c, old_p))

    out = args.path if args.in_place else (args.out or
                                           os.path.splitext(args.path)[0] + ".regraded.jsonl")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)

    n_old = sum(bool(r.get("correct_old")) for r in recs if "output" in r)
    n_new = sum(bool(r.get("correct")) for r in recs if "output" in r)
    print(f"{n_graded} records graded with {GRADER_VERSION}")
    print(f"correct: {n_old} before -> {n_new} after   ({len(flips)} changed)\n")
    for i, r, old_c, old_p in flips:
        cond = r.get("condition") or r.get("arm", "")
        print(f"  #{i:<4} {r.get('unique_id', ''):<40} {cond:<9} seed={r.get('seed')}  "
              f"gold={r['gold']!r}")
        print(f"        old: {old_p!r} {old_c}   new: {r['pred']!r} {r['correct']}  "
              f"boxes={r['preds_all']}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
