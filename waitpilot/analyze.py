"""
analyze.py - effect estimates from branch runs (runs locally, no GPU).

    python -m waitpilot.analyze --forks runs/pilot/pilot_forks.jsonl,runs/pilot/pilot_placebo.jsonl \
        --branches runs/pilot/pilot_branches.jsonl --branch-probes runs/pilot/pilot_branch_probes.jsonl \
        --out-dir runs/pilot/analysis --label exploratory

tau for a fork = mean(outcome | ban arm) - mean(outcome | free arm) over its K continuations.
Length (total tokens) is the primary outcome; accuracy is secondary.

Report sections:
  1. Primary: slope of the length effect (rest-of-trace ban) on state s, adjusting for
     fork position (log prefix length). Cluster bootstrap over problems.
  2. Groups: mean effect for s below / at / above 0.5, and the below-minus-above contrast.
  3. Later markers only (first marker excluded): the same contrast and slope.
  4. Marker vs placebo: one-token ban of the marker minus one-token ban of a matched
     ordinary word. Separates a marker-specific effect from perturbation.
  5. Absorption: does the re-check survive the ban (via substitute phrases)?
  6. Spread: is the variance of tau across forks larger than sampling noise alone?
  7. Secondary outcomes: accuracy, tokens after the first correct answer, re-check.
  8. Reliability of s (split-half over the 4 probes).
Every interval is a 95% percentile interval from resampling whole problems.

Sign convention: tau < 0 for length means the ban shortens the trace.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

CUT = 0.5
OUTCOMES = {                      # name -> branch record field (or derived)
    "length": "n_tokens_total",
    "correct": "correct",
    "recheck": "recheck_follows",
    "after_first_correct": "tokens_after_first_correct",   # from branch probes
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_many(paths):
    out = []
    for p in paths.split(","):
        if p.strip():
            out += load_jsonl(p.strip())
    return out


def group_of(s):
    return "below" if s < CUT else "at" if s == CUT else "above"


# --- per-fork table ------------------------------------------------------------------------

def per_fork_table(forks, branches, probes):
    """One row per fork: features plus, per arm, mean / variance / n of every outcome."""
    after = {p["branch_id"]: p.get("tokens_after_first_correct") for p in probes}
    by_fork = defaultdict(lambda: defaultdict(list))
    for b in branches:
        arm = "free" if b["arm"] == "free" else f"ban_{b['window']}"
        b = dict(b)
        b["tokens_after_first_correct"] = after.get(b["branch_id"])
        by_fork[b["fork_id"]][arm].append(b)

    rows = []
    for f in forks:
        arms = by_fork.get(f["fork_id"])
        if not arms or "free" not in arms or f.get("s") is None:
            continue
        row = {
            "fork_id": f["fork_id"], "unique_id": f["unique_id"], "seed": f["seed"],
            "level": f["level"], "s": f["s"], "group": group_of(f["s"]),
            "pos": f["prefix_len_output"], "log_pos": math.log(max(1, f["prefix_len_output"])),
            "is_first_marker": bool(f.get("is_first_marker")),
            "is_placebo": bool(f.get("is_placebo")),
            "matched_fork_id": f.get("matched_fork_id"),
            "marker_kind": f.get("marker_kind"), "p_marker": f.get("p_marker"),
            "arms": {},
        }
        for arm, bs in arms.items():
            stats = {"n": len(bs),
                     "truncated": sum(bool(b.get("truncated")) for b in bs),
                     "subs_per_1k": 1000 * sum(sum(b["substitute_counts"].values()) for b in bs)
                     / max(1, sum(b["n_tokens_continuation"] for b in bs))}
            for name, field in OUTCOMES.items():
                vals = [float(b[field]) for b in bs if b.get(field) is not None]
                stats[name] = (float(np.mean(vals)) if vals else None,
                               float(np.var(vals, ddof=1)) if len(vals) > 1 else None,
                               len(vals))
            row["arms"][arm] = stats
        rows.append(row)
    return rows


def tau(row, outcome, arm):
    """(effect, sampling variance) of `arm` vs free for one fork, or None."""
    a, f = row["arms"].get(arm), row["arms"].get("free")
    if not a or not f or a[outcome][0] is None or f[outcome][0] is None:
        return None
    (ma, va, na), (mf, vf, nf) = a[outcome], f[outcome]
    var = ((va or 0) / max(na, 1)) + ((vf or 0) / max(nf, 1))
    return ma - mf, var


# --- estimators ------------------------------------------------------------------------------

def ols(y, X, w=None):
    X = np.column_stack([np.ones(len(y))] + list(X))
    if w is not None:
        sw = np.sqrt(np.asarray(w))
        coef, *_ = np.linalg.lstsq(X * sw[:, None], np.asarray(y) * sw, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
    return coef


class Boot:
    """Cluster bootstrap over problems: resample unique_id with replacement."""

    def __init__(self, n_boot, seed=0):
        self.n_boot, self.rng = n_boot, np.random.default_rng(seed)

    def ci(self, units, stat):
        """units: list of (cluster_id, item). stat: list of items -> float or None."""
        clusters = defaultdict(list)
        for cid, item in units:
            clusters[cid].append(item)
        ids = list(clusters)
        point = stat([it for _, it in units])
        if point is None or len(ids) < 2:
            return point, None, None
        vals = []
        for _ in range(self.n_boot):
            pick = self.rng.integers(0, len(ids), len(ids))
            sample = [it for i in pick for it in clusters[ids[i]]]
            v = stat(sample)
            if v is not None and np.isfinite(v):
                vals.append(v)
        if len(vals) < 50:
            return point, None, None
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return point, float(lo), float(hi)


def fmt(est, unit=""):
    point, lo, hi = est
    if point is None:
        return "n/a"
    if lo is None:
        return f"{point:+.3g}{unit} (no CI)"
    return f"{point:+.3g}{unit}  [{lo:+.3g}, {hi:+.3g}]"


def slope_stat(outcome, arm, use_pos=True, weighted=False):
    def stat(rows):
        pts = [(tau(r, outcome, arm), r) for r in rows]
        pts = [(t[0], r) for t, r in pts if t is not None]
        if len(pts) < 5 or len({r["s"] for _, r in pts}) < 2:
            return None
        y = [t for t, _ in pts]
        X = [[r["s"] for _, r in pts]]
        if use_pos:
            X.append([r["log_pos"] for _, r in pts])
        w = None
        if weighted:          # each problem counts equally
            cnt = defaultdict(int)
            for _, r in pts:
                cnt[r["unique_id"]] += 1
            w = [1 / cnt[r["unique_id"]] for _, r in pts]
        return float(ols(y, X, w)[1])
    return stat


def mean_stat(outcome, arm):
    def stat(rows):
        ts = [tau(r, outcome, arm) for r in rows]
        ts = [t[0] for t in ts if t is not None]
        return float(np.mean(ts)) if ts else None
    return stat


def contrast_stat(outcome, arm):
    def stat(rows):
        lo = [tau(r, outcome, arm) for r in rows if r["group"] == "below"]
        hi = [tau(r, outcome, arm) for r in rows if r["group"] == "above"]
        lo = [t[0] for t in lo if t is not None]
        hi = [t[0] for t in hi if t is not None]
        return float(np.mean(lo) - np.mean(hi)) if lo and hi else None
    return stat


# --- report ----------------------------------------------------------------------------------

def report(rows, boot, label, out_dir):
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    markers = [r for r in rows if not r["is_placebo"]]
    placebos = {r["matched_fork_id"]: r for r in rows if r["is_placebo"]}
    units = [(r["unique_id"], r) for r in markers]
    summary = {"label": label, "n_marker_forks": len(markers),
               "n_placebo_forks": len(placebos),
               "n_problems": len({r["unique_id"] for r in markers})}

    p(f"=== analyze.py  ({label}) ===")
    p(f"{len(markers)} marker forks with branches, {len(placebos)} placebo forks, "
      f"{summary['n_problems']} problems. Intervals: 95%, cluster bootstrap over problems "
      f"({boot.n_boot} resamples).")
    arms_seen = sorted({a for r in markers for a in r["arms"]})
    p(f"arms: {arms_seen}")
    trunc = sum(r["arms"][a]["truncated"] for r in rows for a in r["arms"])
    p(f"continuations at the token cap: {trunc} (counted as wrong for accuracy; their "
      f"length is the cap)")

    p("\n1. PRIMARY  length effect of the rest-of-trace ban vs state s")
    est = boot.ci(units, slope_stat("length", "ban_rest"))
    summary["primary_slope_length"] = est
    p(f"   slope of tau on s, adjusted for log position:   {fmt(est, ' tok')}")
    p(f"   without position adjustment:                    "
      f"{fmt(boot.ci(units, slope_stat('length', 'ban_rest', use_pos=False)), ' tok')}")
    p(f"   each problem weighted equally (check):          "
      f"{fmt(boot.ci(units, slope_stat('length', 'ban_rest', weighted=True)), ' tok')}")
    p(f"   mean length effect, all forks:                  "
      f"{fmt(boot.ci(units, mean_stat('length', 'ban_rest')), ' tok')}")
    p("   (H1 predicts the ban shortens traces more when the answer is already right:")
    p("    a negative slope for length.)")

    p("\n2. GROUPS  mean length effect (rest-of-trace ban) by state")
    for g in ("below", "at", "above"):
        sub = [(u, r) for u, r in units if r["group"] == g]
        est = boot.ci(sub, mean_stat("length", "ban_rest"))
        summary[f"group_{g}_length"] = est
        p(f"   s {g:<6} n={len(sub):>4}   {fmt(est, ' tok')}")
    est = boot.ci(units, contrast_stat("length", "ban_rest"))
    summary["contrast_below_minus_above_length"] = est
    p(f"   below minus above:        {fmt(est, ' tok')}")
    p("   (the 'at 0.5' group is reported, not interpreted)")

    p("\n3. LATER MARKERS ONLY  (first marker in each trace excluded)")
    later = [(u, r) for u, r in units if not r["is_first_marker"]]
    est_c = boot.ci(later, contrast_stat("length", "ban_rest"))
    est_s = boot.ci(later, slope_stat("length", "ban_rest"))
    summary["later_contrast_length"], summary["later_slope_length"] = est_c, est_s
    p(f"   n={len(later)}   below minus above: {fmt(est_c, ' tok')}")
    p(f"            slope on s (+position): {fmt(est_s, ' tok')}")

    p("\n4. MARKER vs PLACEBO  one-token ban, marker minus matched ordinary word")
    pairs = [(r["unique_id"], (r, placebos[r["fork_id"]])) for r in markers
             if r["fork_id"] in placebos]
    if not pairs:
        p("   no placebo forks with branches")
    for outcome, unit in (("length", " tok"), ("recheck", ""), ("correct", "")):
        def diff_stat(items, outcome=outcome):
            d = []
            for m, pl in items:
                tm, tp = tau(m, outcome, "ban_token"), tau(pl, outcome, "ban_token")
                if tm is not None and tp is not None:
                    d.append(tm[0] - tp[0])
            return float(np.mean(d)) if d else None

        def side(items, which, outcome=outcome):
            ts = [tau(x[which], outcome, "ban_token") for x in items]
            ts = [t[0] for t in ts if t is not None]
            return float(np.mean(ts)) if ts else None
        if pairs:
            est = boot.ci(pairs, diff_stat)
            summary[f"marker_minus_placebo_{outcome}"] = est
            p(f"   {outcome:<8} marker {side([x for _, x in pairs], 0):+.3g}{unit}   "
              f"placebo {side([x for _, x in pairs], 1):+.3g}{unit}   "
              f"difference {fmt(est, unit)}")
    if pairs:
        for g in ("below", "above"):
            sub = [(u, x) for u, x in pairs if x[0]["group"] == g]

            def dstat(items):
                d = []
                for m, pl in items:
                    tm, tp = tau(m, "length", "ban_token"), tau(pl, "length", "ban_token")
                    if tm is not None and tp is not None:
                        d.append(tm[0] - tp[0])
                return float(np.mean(d)) if d else None
            p(f"   length difference, marker state {g:<5} n={len(sub):>4}: "
              f"{fmt(boot.ci(sub, dstat), ' tok')}")
        gaps = [placebos[r["fork_id"]].get("match_p_gap") for r in markers
                if r["fork_id"] in placebos]
        gaps = [g for g in gaps if g is not None]
        if gaps:
            p(f"   placebo match quality: median |p_word - p_marker| = "
              f"{np.median(gaps):.2f} (max {max(gaps):.2f})")

    p("\n5. ABSORPTION  re-check within the first 300 characters after the fork")
    for arm in ("free", "ban_token", "ban_sentence", "ban_rest"):
        rs = [r["arms"][arm]["recheck"][0] for r in markers if arm in r["arms"]
              and r["arms"][arm]["recheck"][0] is not None]
        subs = [r["arms"][arm]["subs_per_1k"] for r in markers if arm in r["arms"]]
        if rs:
            p(f"   {arm:<13} re-check rate {np.mean(rs):.2f}   substitute phrases per "
              f"1k tokens {np.mean(subs):.2f}   (forks: {len(rs)})")
    free_rate = np.mean([r["arms"]["free"]["recheck"][0] for r in markers])
    rest = [r["arms"]["ban_rest"]["recheck"][0] for r in markers if "ban_rest" in r["arms"]]
    if rest and free_rate > 0:
        ratio = np.mean(rest) / free_rate
        summary["absorption_ratio_rest"] = float(ratio)
        p(f"   absorption ratio (ban_rest / free re-check rate): {ratio:.2f}  "
          f"(1 = the re-check fully survives the ban = stylistic; 0 = it disappears)")

    p("\n6. SPREAD  is the variation in effects across forks more than sampling noise?")
    for outcome in ("length", "correct"):
        ts = [tau(r, outcome, "ban_rest") for r in markers]
        ts = [t for t in ts if t is not None]
        if len(ts) > 2:
            obs = float(np.var([t[0] for t in ts], ddof=1))
            noise = float(np.mean([t[1] for t in ts]))
            share = max(0.0, 1 - noise / obs) if obs > 0 else 0.0
            summary[f"spread_{outcome}"] = {"observed_var": obs, "noise_var": noise,
                                            "excess_share": share}
            p(f"   {outcome:<8} observed variance {obs:.4g}, expected from noise "
              f"{noise:.4g}  -> share beyond noise {share:.0%}")

    p("\n7. SECONDARY OUTCOMES  (rest-of-trace ban)")
    for outcome, unit in (("correct", ""), ("after_first_correct", " tok"), ("recheck", "")):
        est = boot.ci(units, mean_stat(outcome, "ban_rest"))
        con = boot.ci(units, contrast_stat(outcome, "ban_rest"))
        summary[f"secondary_{outcome}"] = {"mean": est, "contrast": con}
        p(f"   {outcome:<20} mean {fmt(est, unit)}   below-above {fmt(con, unit)}")
    n_after = defaultdict(lambda: [0, 0])
    for r in markers:
        for arm, st in r["arms"].items():
            n_after[arm][0] += st["after_first_correct"][2]
            n_after[arm][1] += st["n"]
    if not any(k for k, _ in n_after.values()):
        p("   (tokens after the first correct answer: no --branch-probes given)")
    else:
        p("   continuations that reach a correct answer (tokens-after-first-correct "
          "defined): " + ", ".join(f"{a} {k}/{n}" for a, (k, n) in sorted(n_after.items())))

    p("\n8. RELIABILITY OF s  (split-half over the probes, Spearman-Brown corrected)")
    rel = split_half(markers_raw=[r for r in rows if not r["is_placebo"]])
    summary["s_reliability"] = rel
    p(f"   {rel:.2f}" if rel is not None else "   n/a (probe-level data missing)")

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


_PROBES = {}


def split_half(markers_raw):
    halves = []
    for r in markers_raw:
        pc = _PROBES.get(r["fork_id"])
        if pc and len(pc) >= 4:
            halves.append((np.mean(pc[0::2]), np.mean(pc[1::2])))
    if len(halves) < 10:
        return None
    a, b = np.array(halves).T
    if a.std() == 0 or b.std() == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return 2 * r / (1 + r)


# --- outputs ---------------------------------------------------------------------------------

def write_csv(rows, path):
    arms = sorted({a for r in rows for a in r["arms"]})
    cols = ["fork_id", "unique_id", "seed", "level", "is_placebo", "matched_fork_id",
            "marker_kind", "is_first_marker", "pos", "s", "group", "p_marker"]
    extra = [f"{a}_{o}" for a in arms for o in OUTCOMES] + [f"tau_{o}_{a}" for a in arms
                                                            if a != "free" for o in OUTCOMES]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + extra)
        for r in rows:
            vals = [r[c] for c in cols]
            for a in arms:
                for o in OUTCOMES:
                    st = r["arms"].get(a)
                    vals.append(st[o][0] if st else None)
            for a in arms:
                if a == "free":
                    continue
                for o in OUTCOMES:
                    t = tau(r, o, a)
                    vals.append(t[0] if t else None)
            w.writerow(vals)


def plots(rows, out_dir, label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: no figures)")
        return
    rng = np.random.default_rng(1)
    markers = [r for r in rows if not r["is_placebo"]]
    placebos = {r["matched_fork_id"]: r for r in rows if r["is_placebo"]}

    # Figure 1: length effect vs state, rest-of-trace ban
    pts = [(r["s"], tau(r, "length", "ban_rest")) for r in markers]
    pts = [(s, t[0]) for s, t in pts if t is not None]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = np.array([s for s, _ in pts]) + rng.uniform(-0.03, 0.03, len(pts))
        ax.scatter(xs, [t for _, t in pts], s=10, alpha=0.4)
        for g, x in (("below", 0.1), ("above", 0.9)):
            ts = [t for s, t in pts if group_of(s) == g]
            if ts:
                ax.errorbar([x], [np.mean(ts)], yerr=[[1.96 * np.std(ts) / math.sqrt(len(ts))]],
                            fmt="o", color="black", capsize=4)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("state s at the fork (share of probes correct)")
        ax.set_ylabel("length effect of ban (tokens, ban - free)")
        ax.set_title(f"Rest-of-trace ban: length effect by state ({label})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_length_vs_state.png"), dpi=150)
        plt.close(fig)

    # Figure 2: marker vs placebo, one-token ban, by state group
    groups, mk, pl = [], [], []
    for g in ("below", "above"):
        m = [tau(r, "length", "ban_token") for r in markers
             if r["group"] == g and r["fork_id"] in placebos]
        q = [tau(placebos[r["fork_id"]], "length", "ban_token") for r in markers
             if r["group"] == g and r["fork_id"] in placebos]
        m = [t[0] for t in m if t is not None]
        q = [t[0] for t in q if t is not None]
        if m and q:
            groups.append(g)
            mk.append((np.mean(m), 1.96 * np.std(m) / math.sqrt(len(m))))
            pl.append((np.mean(q), 1.96 * np.std(q) / math.sqrt(len(q))))
    if groups:
        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(len(groups))
        ax.bar(x - 0.18, [v for v, _ in mk], 0.36, yerr=[e for _, e in mk], capsize=4,
               label="marker")
        ax.bar(x + 0.18, [v for v, _ in pl], 0.36, yerr=[e for _, e in pl], capsize=4,
               label="placebo word")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x, [f"s {g} 0.5" for g in groups])
        ax.set_ylabel("length effect of one-token ban (tokens)")
        ax.set_title(f"Marker vs placebo ({label})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_marker_vs_placebo.png"), dpi=150)
        plt.close(fig)

    # Figure 3: re-check rate by arm (absorption)
    arms = [a for a in ("free", "ban_token", "ban_sentence", "ban_rest")
            if any(a in r["arms"] for r in markers)]
    rates = [np.mean([r["arms"][a]["recheck"][0] for r in markers if a in r["arms"]])
             for a in arms]
    if arms:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(arms, rates)
        ax.set_ylim(0, 1)
        ax.set_ylabel("re-check within 300 chars of the fork")
        ax.set_title(f"Absorption by ban window ({label})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_absorption.png"), dpi=150)
        plt.close(fig)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--forks", required=True, help="fork file(s), comma-separated")
    ap.add_argument("--branches", required=True, help="branch file(s), comma-separated")
    ap.add_argument("--branch-probes", default="", help="branch probe file(s), optional")
    ap.add_argument("--out-dir", default="runs/main/analysis")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--label", default="exploratory",
                    help="printed on every output; pilot results are exploratory")
    args = ap.parse_args()

    forks = load_many(args.forks)
    branches = load_many(args.branches)
    probes = load_many(args.branch_probes) if args.branch_probes else []
    for f in forks:
        _PROBES[f["fork_id"]] = f.get("probe_correct")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = per_fork_table(forks, branches, probes)
    if not rows:
        sys.exit("no forks with branches: check that --forks and --branches match")
    report(rows, Boot(args.boot), args.label, args.out_dir)
    write_csv(rows, os.path.join(args.out_dir, "per_fork.csv"))
    plots(rows, args.out_dir, args.label)
    print(f"\nwritten to {args.out_dir}: report.txt, summary.json, per_fork.csv, figures")


if __name__ == "__main__":
    main()
