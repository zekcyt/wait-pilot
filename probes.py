"""
probes.py - forced-answer probes (design doc, Measurement).

A probe appends PROBE_SUFFIX ("Final Answer: \\boxed{") to a token-ID prefix and samples
a short answer. The share of correct probe answers at a fork is the state s.

  forks      Pick forks in each baseline trace (the first marker plus a random sample of
             later ones) and record per-fork features: marker type and token IDs,
             paragraph-initial, p_marker, prefix length, state s from --probes samples,
             and whether the next sentence changes the probed answer.
             This is all the threshold pilot batch needs (no branches).
  threshold  Count forks with s below, at and above the cut. Looks at fork counts only,
             never at effects, so it can run before the threshold is locked.
  branch     Probe along each branch at paragraph breaks: where the first correct answer
             appears, tokens after it, and whether a wrong state was recovered.

    python probes.py forks --baselines runs/pilot_baselines.jsonl --out runs/pilot_forks.jsonl
    python probes.py threshold runs/pilot_forks.jsonl
    python probes.py branch --branches runs/branches.jsonl --forks runs/forks.jsonl --out runs/branch_probes.jsonl
"""
import argparse
import math
import random
import sys
import time
from collections import Counter, defaultdict

from common import (DEFAULT_MODEL, PROBE_MAX_TOKENS, PROBE_SUFFIX, THINK_END, MarkerBan,
                    find_marker_positions, next_sentence_end, paragraph_break_ids)
from grading import answers_equal, grade_answers, probe_answer

from cloud_generate import (Engine, Req, append_jsonl, chunks, load_jsonl, load_tokenizer,
                            stable_seed, stamp)


def select_forks(markers, n_forks, key):
    """The first marker plus a deterministic random sample of the later ones."""
    if not markers:
        return []
    rest = markers[1:]
    rng = random.Random(key)
    picked = rng.sample(rest, min(len(rest), max(0, n_forks - 1)))
    return [markers[0]] + sorted(picked, key=lambda m: m["pos"])


def modal_answer(answers):
    """Most common answer, with equivalent forms grouped. None if no answer parsed."""
    groups = []          # [representative, count]
    for a in answers:
        if a is None:
            continue
        for g in groups:
            if answers_equal(a, g[0]):
                g[1] += 1
                break
        else:
            groups.append([a, 1])
    return max(groups, key=lambda g: g[1])[0] if groups else None


def probe_reqs(prefix, n, key, suffix_ids, tag):
    return [Req(prefix + suffix_ids, PROBE_MAX_TOKENS, stable_seed("probe", key, j),
                tag=(tag, j)) for j in range(n)]


def score_probes(tok, results, gold):
    answers = [probe_answer(tok.decode(r.token_ids, skip_special_tokens=True))
               for r in results]
    correct = [bool(a) and grade_answers([a], gold)[0] for a in answers]
    s = sum(correct) / len(correct) if correct else None
    return answers, correct, s


# --- forks -------------------------------------------------------------------------------

def run_forks(engine, tok, bases, out, n_forks, n_probes, chunk):
    ban = engine.ban
    para = paragraph_break_ids(tok)
    end_id = tok.convert_tokens_to_ids(THINK_END)
    suffix_ids = tok.encode(PROBE_SUFFIX, add_special_tokens=False)
    single = set(ban.single_ids)
    done = {r["fork_id"] for r in load_jsonl(out)}

    jobs = []
    for b in bases:
        markers = find_marker_positions(tok, ban, b["output_token_ids"], para, end_id)
        chosen = select_forks(markers, n_forks, f"{b['unique_id']}|{b['seed']}")
        for rank, m in enumerate(chosen):
            fid = f"{b['unique_id']}|s{b['seed']}|t{m['pos']}"
            if fid not in done:
                jobs.append((b, m, fid, markers.index(m), len(markers)))
    print(f"{len(bases)} baseline traces: {len(jobs)} forks to probe, {len(done)} already "
          f"in {out}")

    for part in chunks(jobs, chunk):
        t0 = time.time()
        reqs = []
        for b, m, fid, idx, n_markers in part:
            ids = b["output_token_ids"]
            prefix = b["prompt_token_ids"] + ids[:m["pos"]]
            sent_end = next_sentence_end(tok, ids, m["pos"])
            after = b["prompt_token_ids"] + ids[:sent_end]
            # p_marker: one step from the prefix with top-20 logprobs (raw, T=1).
            reqs.append(Req(prefix, 1, stable_seed("pw", fid), logprobs=20,
                            tag=((fid, "pw"), 0)))
            reqs += probe_reqs(prefix, n_probes, fid, suffix_ids, (fid, "fork"))
            reqs += probe_reqs(after, n_probes, fid + "|next", suffix_ids, (fid, "next"))
        by = defaultdict(list)
        for res in engine.run(reqs):
            by[res.req.tag[0]].append(res)

        recs = []
        for b, m, fid, idx, n_markers in part:
            pw = by[(fid, "pw")][0]
            lp = pw.logprobs[0] if pw.logprobs else {}
            p_marker = sum(math.exp(v) for t, v in lp.items() if t in single)
            a_f, c_f, s = score_probes(tok, by[(fid, "fork")], b["gold"])
            a_n, c_n, s_next = score_probes(tok, by[(fid, "next")], b["gold"])
            mod_f, mod_n = modal_answer(a_f), modal_answer(a_n)
            recs.append({
                "fork_id": fid, "unique_id": b["unique_id"], "seed": b["seed"],
                "level": b["level"], "subject": b.get("subject"), "gold": b["gold"],
                "pos": m["pos"], "prefix_len_output": m["pos"],
                "prefix_len_total": len(b["prompt_token_ids"]) + m["pos"],
                "marker_text": m["text"], "marker_kind": m["kind"],
                "marker_token_ids": m["token_ids"], "paragraph_initial": m["paragraph_initial"],
                "marker_index": idx, "is_first_marker": idx == 0,
                "n_markers_in_trace": n_markers,
                "p_marker": p_marker,
                "p_marker_note": "sum of raw (T=1) probs of single marker IDs in top-20",
                "s": s, "probe_answers": a_f, "probe_correct": c_f,
                "modal_answer": mod_f,
                "modal_correct": bool(mod_f) and grade_answers([mod_f], b["gold"])[0],
                "s_next_sentence": s_next, "probe_answers_next": a_n,
                "answer_changed_next_sentence": (
                    None if mod_f is None and mod_n is None
                    else not (mod_f and mod_n and answers_equal(mod_n, mod_f))),
                "n_probes": len(c_f), "probe_suffix": PROBE_SUFFIX,
                "baseline_correct": b["correct"], "baseline_truncated": b["truncated"],
                "baseline_truncation_phase": b.get("truncation_phase"),
                "baseline_n_tokens": b["n_tokens"],
                "engine": f"vllm {engine.vllm_version}", **stamp(),
            })
        append_jsonl(out, recs)
        print(f"  chunk: {len(recs)} forks probed in {time.time() - t0:.0f}s")


# --- threshold report -----------------------------------------------------------------------

def threshold_report(path, cut, min_share):
    forks = [f for f in load_jsonl(path) if f.get("s") is not None]
    if not forks:
        sys.exit(f"no probed forks in {path}")

    def groups(fs):
        c = Counter("below" if f["s"] < cut else "at" if f["s"] == cut else "above"
                    for f in fs)
        return c["below"], c["at"], c["above"]

    def line(name, fs):
        lo, at, hi = groups(fs)
        n = len(fs)
        if not n:
            return
        print(f"  {name:<22}{n:>6}{lo:>8} ({lo / n:>4.0%}){at:>8} ({at / n:>4.0%})"
              f"{hi:>8} ({hi / n:>4.0%})")

    print(f"{len(forks)} forks in {path}; cut = {cut}\n")
    print(f"  {'group':<22}{'forks':>6}{'below':>15}{'at':>15}{'above':>15}")
    line("all", forks)
    for lv in sorted({f["level"] for f in forks}):
        line(f"level {lv}", [f for f in forks if f["level"] == lv])
    line("first marker", [f for f in forks if f["is_first_marker"]])
    line("later markers", [f for f in forks if not f["is_first_marker"]])
    line("baseline truncated", [f for f in forks if f["baseline_truncated"]])

    vals = Counter(round(f["s"], 3) for f in forks)
    print("\n  distribution of s: " + "  ".join(f"{k}: {v}" for k, v in sorted(vals.items())))
    lo, at, hi = groups(forks)
    small = [g for g, n in (("below", lo), ("above", hi)) if n / len(forks) < min_share]
    print(f"\n  per-problem: {len({f['unique_id'] for f in forks})} problems")
    if small:
        print(f"\n  WARNING: group(s) {small} hold less than {min_share:.0%} of forks. "
              f"Per the preregistration the cut may move once, before v2 is tagged.")
    else:
        print(f"\n  below and above both hold at least {min_share:.0%} of forks.")
    print("\n  (Counts only. No effect estimates are computed here.)")


# --- branch probes --------------------------------------------------------------------------

def run_branch_probes(engine, tok, branches, forks, bases, out, n_probes, max_breaks, chunk):
    para = set(paragraph_break_ids(tok))
    end_id = tok.convert_tokens_to_ids(THINK_END)
    suffix_ids = tok.encode(PROBE_SUFFIX, add_special_tokens=False)
    done = {r["branch_id"] for r in load_jsonl(out)}
    todo = [b for b in branches if b["branch_id"] not in done]
    print(f"{len(branches)} branches: {len(todo)} to probe, {len(done)} already in {out}")

    for part in chunks(todo, chunk):
        t0 = time.time()
        reqs, breaks_of = [], {}
        for br in part:
            f = forks[br["fork_id"]]
            base = bases[(br["unique_id"], br["seed"])]
            cont = br["continuation_token_ids"]
            end = cont.index(end_id) if end_id in cont else len(cont)
            brks = [i + 1 for i in range(end) if cont[i] in para]
            if len(brks) > max_breaks:          # keep an even spread, always the last
                step = len(brks) / max_breaks
                brks = sorted({brks[int(j * step)] for j in range(max_breaks)} | {brks[-1]})
            breaks_of[br["branch_id"]] = brks
            prefix = base["prompt_token_ids"] + base["output_token_ids"][:f["pos"]]
            for b in brks:
                reqs += probe_reqs(prefix + cont[:b], n_probes, f"{br['branch_id']}|{b}",
                                   suffix_ids, (br["branch_id"], b))
        by = defaultdict(list)
        for res in engine.run(reqs):
            by[res.req.tag[0]].append(res)

        recs = []
        for br in part:
            f = forks[br["fork_id"]]
            bid = br["branch_id"]
            fork_correct = f.get("modal_correct", False)
            first, trail = (0 if fork_correct else None), []
            for b in breaks_of[bid]:
                _, c, s = score_probes(tok, by[(bid, b)], br["gold"])
                trail.append([b, s])
                if first is None and s is not None and s >= 0.5:
                    first = b
            n_cont = br["n_tokens_continuation"]
            recs.append({
                "branch_id": bid, "fork_id": br["fork_id"], "arm": br["arm"],
                "window": br["window"], "k": br["k"], "unique_id": br["unique_id"],
                "seed": br["seed"], "fork_s": f.get("s"), "fork_modal_correct": fork_correct,
                "first_correct_pos": first,           # tokens after the fork; 0 = at fork
                "tokens_after_first_correct": (n_cont - first) if first is not None else None,
                "recovered": (not fork_correct) and bool(br["correct"]),
                "final_correct": br["correct"],
                "probe_trail": trail,                 # [break position, share correct]
                "n_probes_per_break": n_probes, **stamp(),
            })
        append_jsonl(out, recs)
        print(f"  chunk: {len(recs)} branches probed in {time.time() - t0:.0f}s")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def engine_args(sp):
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--ban-impl", choices=["ids", "words"], default="ids")
        sp.add_argument("--gpu-mem", type=float, default=0.92)
        sp.add_argument("--chunk", type=int, default=256)

    fk = sub.add_parser("forks")
    engine_args(fk)
    fk.add_argument("--baselines", required=True)
    fk.add_argument("--forks-per-trace", type=int, default=3,
                    help="first marker + (n-1) random later ones. TBD in preregistration")
    fk.add_argument("--probes", type=int, default=4,
                    help="probe samples per fork. TBD in preregistration")
    fk.add_argument("--out", default="runs/forks.jsonl")

    th = sub.add_parser("threshold")
    th.add_argument("path")
    th.add_argument("--cut", type=float, default=0.5)
    th.add_argument("--min-share", type=float, default=0.15,
                    help="warn if below/above holds less than this share. TBD")

    bp = sub.add_parser("branch")
    engine_args(bp)
    bp.add_argument("--branches", required=True)
    bp.add_argument("--forks", required=True)
    bp.add_argument("--baselines", required=True)
    bp.add_argument("--probes", type=int, default=1, help="probe samples per break")
    bp.add_argument("--max-breaks", type=int, default=48)
    bp.add_argument("--out", default="runs/branch_probes.jsonl")
    args = p.parse_args()

    if args.cmd == "threshold":
        threshold_report(args.path, args.cut, args.min_share)
        return

    tok = load_tokenizer(args.model)
    engine = Engine(tok, MarkerBan(tok), args.model, args.ban_impl,
                    gpu_memory_utilization=args.gpu_mem)
    if args.cmd == "forks":
        run_forks(engine, tok, load_jsonl(args.baselines), args.out, args.forks_per_trace,
                  args.probes, args.chunk)
    else:
        forks = {f["fork_id"]: f for f in load_jsonl(args.forks)}
        bases = {(r["unique_id"], r["seed"]): r for r in load_jsonl(args.baselines)}
        run_branch_probes(engine, tok, load_jsonl(args.branches), forks, bases, args.out,
                          args.probes, args.max_breaks, args.chunk)


if __name__ == "__main__":
    main()
