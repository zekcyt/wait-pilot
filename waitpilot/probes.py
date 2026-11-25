"""
probes.py - forced-answer probes (design doc, Measurement).

A probe appends PROBE_SUFFIX ("Final Answer: \\boxed{") to a token-ID prefix and samples
a short answer. The share of correct probe answers at a fork is the state s.

  forks      Pick forks in each baseline trace (the first marker plus a random sample of
             later ones) and record per-fork features: marker type and token IDs,
             paragraph-initial, p_marker, prefix length, state s from --probes samples,
             and whether the next sentence changes the probed answer.
             This is all the threshold pilot batch needs (no branches).
  placebo    For each marker fork, a matched placebo fork: a nearby ordinary word
             ("So", "Let", "The" ...) whose probability at its position is closest to the
             marker's p_marker, measured with the same probes. Banning it controls for the
             effect of banning *any* likely token (perturbation vs marker-specific).
  threshold  Count forks with s below, at and above the cut. Looks at fork counts only,
             never at effects, so it can run before the threshold is locked.
  branch     Probe along each branch at paragraph breaks (every --probe-every'th break):
             where the first correct answer appears, tokens after it, and whether a wrong
             state was recovered.

    python -m waitpilot.probes forks --baselines runs/pilot/pilot_baselines.jsonl --out runs/pilot/pilot_forks.jsonl
    python -m waitpilot.probes placebo --baselines runs/pilot/pilot_baselines.jsonl --forks runs/pilot/pilot_forks.jsonl --out runs/pilot/pilot_placebo.jsonl
    python -m waitpilot.probes threshold runs/pilot/pilot_forks.jsonl
    python -m waitpilot.probes branch --branches runs/main/branches.jsonl --forks runs/main/forks.jsonl,runs/main/placebo.jsonl --baselines runs/main/baselines.jsonl --out runs/main/branch_probes.jsonl --probe-every 2
"""
import argparse
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

from waitpilot.common import (DEFAULT_MODEL, MARKER_RE, PROBE_MAX_TOKENS, PROBE_SUFFIX, THINK_END,
                    MarkerBan, find_marker_positions, next_sentence_end,
                    paragraph_break_ids)
from waitpilot.grading import answers_equal, grade_answers, probe_answer

from waitpilot.cloud_generate import (Engine, Req, append_jsonl, chunks, load_input, load_inputs,
                            load_jsonl,
                            load_tokenizer,
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

def probe_points(engine, tok, jobs, out, n_probes, chunk):
    """Measure fork features at token positions in baseline traces and append one record
    per job. Shared by marker forks and placebo forks, so both are measured the same way.

    job: dict with b (baseline), pos, text, kind, token_ids, paragraph_initial, fid,
         p_ids (token IDs whose probability is summed into p_marker), extra (record fields).
    """
    suffix_ids = tok.encode(PROBE_SUFFIX, add_special_tokens=False)
    for part in chunks(jobs, chunk):
        t0 = time.time()
        reqs = []
        for j in part:
            b, fid = j["b"], j["fid"]
            ids = b["output_token_ids"]
            prefix = b["prompt_token_ids"] + ids[:j["pos"]]
            sent_end = next_sentence_end(tok, ids, j["pos"])
            after = b["prompt_token_ids"] + ids[:sent_end]
            # p_marker: one step from the prefix with top-20 logprobs (raw, T=1).
            reqs.append(Req(prefix, 1, stable_seed("pw", fid), logprobs=20,
                            tag=((fid, "pw"), 0)))
            reqs += probe_reqs(prefix, n_probes, fid, suffix_ids, (fid, "fork"))
            reqs += probe_reqs(after, n_probes, fid + "|next", suffix_ids, (fid, "next"))
        by = defaultdict(list)
        for res in engine.run(reqs, stage="fork_probes"):
            by[res.req.tag[0]].append(res)

        recs = []
        for j in part:
            b, fid = j["b"], j["fid"]
            pw = by[(fid, "pw")][0]
            lp = pw.logprobs[0] if pw.logprobs else {}
            p_marker = sum(math.exp(v) for t, v in lp.items() if t in j["p_ids"])
            a_f, c_f, s = score_probes(tok, by[(fid, "fork")], b["gold"])
            a_n, c_n, s_next = score_probes(tok, by[(fid, "next")], b["gold"])
            mod_f, mod_n = modal_answer(a_f), modal_answer(a_n)
            recs.append({
                "fork_id": fid, "unique_id": b["unique_id"], "seed": b["seed"],
                "level": b["level"], "subject": b.get("subject"), "gold": b["gold"],
                "pos": j["pos"], "prefix_len_output": j["pos"],
                "prefix_len_total": len(b["prompt_token_ids"]) + j["pos"],
                "marker_text": j["text"], "marker_kind": j["kind"],
                "marker_token_ids": j["token_ids"],
                "paragraph_initial": j["paragraph_initial"],
                **j["extra"],
                "p_marker": p_marker,
                "p_marker_note": "sum of raw (T=1) probs of the banned IDs in top-20",
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


def run_forks(engine, tok, bases, out, n_forks, n_probes, chunk):
    ban = engine.ban
    para = paragraph_break_ids(tok)
    end_id = tok.convert_tokens_to_ids(THINK_END)
    single = set(ban.single_ids)
    done = {r["fork_id"] for r in load_jsonl(out)}

    jobs = []
    for b in bases:
        markers = find_marker_positions(tok, ban, b["output_token_ids"], para, end_id)
        chosen = select_forks(markers, n_forks, f"{b['unique_id']}|{b['seed']}")
        for m in chosen:
            fid = f"{b['unique_id']}|s{b['seed']}|t{m['pos']}"
            if fid in done:
                continue
            idx = markers.index(m)
            jobs.append({"b": b, "fid": fid, "pos": m["pos"], "text": m["text"],
                         "kind": m["kind"], "token_ids": m["token_ids"],
                         "paragraph_initial": m["paragraph_initial"], "p_ids": single,
                         "extra": {"is_placebo": False, "marker_index": idx,
                                   "is_first_marker": idx == 0,
                                   "n_markers_in_trace": len(markers)}})
    print(f"{len(bases)} baseline traces: {len(jobs)} forks to probe, {len(done)} already "
          f"in {out}")
    probe_points(engine, tok, jobs, out, n_probes, chunk)


# --- placebo forks ---------------------------------------------------------------------------

PLACEBO_WINDOW = 300        # look for a placebo word within this many tokens of the fork
PLACEBO_MIN_GAP = 5         # ...and at least this far from any marker
PLACEBO_CANDIDATES = 8      # nearest candidates whose probability is measured
_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def word_token_ids(tok):
    """stripped word -> all token IDs that decode to it (" So", "So", "\\nSo" ...).
    Banning all of them mirrors how the marker ban covers spacing variants."""
    by_word = defaultdict(list)
    for tid in range(len(tok)):
        w = tok.decode([tid]).strip()
        if _WORD_RE.fullmatch(w):
            by_word[w].append(tid)
    return by_word


def placebo_candidates(tok, ban, b, fork, markers, by_word, para, end_id):
    """Positions near the fork where the model wrote an ordinary word: letters only, not
    a marker or an "H"-prefix, not within PLACEBO_MIN_GAP tokens of any marker, inside
    the thinking and after the first paragraph break. Nearest first."""
    ids = b["output_token_ids"]
    end = ids.index(end_id) if end_id in ids else len(ids)
    first_para = next((i for i in range(end) if ids[i] in para), end)
    blocked = set(ban.single_ids) | set(ban.seq_prefix_ids) | set(ban.seq_next_ids)
    near_marker = {p + d for m in markers for p in [m["pos"]]
                   for d in range(-PLACEBO_MIN_GAP, PLACEBO_MIN_GAP + 1)}
    lo = max(first_para + 1, fork["pos"] - PLACEBO_WINDOW)
    hi = min(end, fork["pos"] + PLACEBO_WINDOW)
    out = []
    for i in range(lo, hi):
        if i in near_marker or ids[i] in blocked:
            continue
        text = tok.decode([ids[i]])
        w = text.strip()
        if w not in by_word or MARKER_RE.search(w):
            continue
        # A whole word, not a fragment like "ic" in "algebra|ic": it starts after
        # whitespace and the next token does not continue it with more letters.
        starts = text[:1].isspace() or tok.decode([ids[i - 1]])[-1:].isspace()
        nxt = tok.decode([ids[i + 1]]) if i + 1 < len(ids) else " "
        if starts and not nxt[:1].isalpha():
            out.append(i)
    out.sort(key=lambda i: (abs(i - fork["pos"]), i))
    return out[:PLACEBO_CANDIDATES]


def run_placebo(engine, tok, bases, forks, out, n_probes, chunk):
    """One placebo fork per marker fork: the nearby ordinary word whose probability at
    its position is closest to the marker's p_marker. Then the same probes as forks."""
    ban = engine.ban
    para = set(paragraph_break_ids(tok))
    end_id = tok.convert_tokens_to_ids(THINK_END)
    by_word = word_token_ids(tok)
    bmap = {(b["unique_id"], b["seed"]): b for b in bases}
    done = {r["matched_fork_id"] for r in load_jsonl(out) if r.get("matched_fork_id")}
    todo = [f for f in forks if not f.get("is_placebo") and f["fork_id"] not in done]
    print(f"{len(todo)} marker forks need a placebo, {len(done)} already in {out}")

    for part in chunks(todo, chunk):
        # 1. measure p(word) at each candidate position
        reqs, cands = [], {}
        for f in part:
            b = bmap[(f["unique_id"], f["seed"])]
            markers = find_marker_positions(tok, ban, b["output_token_ids"],
                                            list(para), end_id)
            cs = placebo_candidates(tok, ban, b, f, markers, by_word, para, end_id)
            cands[f["fork_id"]] = cs
            for i in cs:
                prefix = b["prompt_token_ids"] + b["output_token_ids"][:i]
                reqs.append(Req(prefix, 1, stable_seed("pc", f["fork_id"], i), logprobs=20,
                                tag=(f["fork_id"], i)))
        lp = {res.req.tag: (res.logprobs[0] if res.logprobs else {})
              for res in engine.run(reqs, stage="placebo_select")}

        # 2. pick the candidate closest in probability to the marker
        jobs, skipped = [], 0
        for f in part:
            b = bmap[(f["unique_id"], f["seed"])]
            best = None
            for i in cands[f["fork_id"]]:
                w = tok.decode([b["output_token_ids"][i]]).strip()
                wids = set(by_word[w])
                p = sum(math.exp(v) for t, v in lp[(f["fork_id"], i)].items() if t in wids)
                if p <= 0:          # not in the top-20: probability unknown, skip
                    continue
                gap = abs(p - f["p_marker"])
                if best is None or gap < best[0]:
                    best = (gap, i, w, wids, p)
            if best is None:
                skipped += 1
                continue
            gap, i, w, wids, p = best
            ids = b["output_token_ids"]
            before = tok.decode(ids[max(0, i - 3):i])
            jobs.append({"b": b, "fid": f"{f['fork_id']}|placebo", "pos": i,
                         "text": tok.decode([ids[i]]), "kind": "placebo",
                         "token_ids": [ids[i]],
                         "paragraph_initial": before.rstrip(" \t").endswith("\n\n"),
                         "p_ids": wids,
                         "extra": {"is_placebo": True, "matched_fork_id": f["fork_id"],
                                   "placebo_word": w, "ban_ids": sorted(wids),
                                   "matched_p_marker": f["p_marker"], "match_p_gap": gap,
                                   "matched_s": f["s"],
                                   "marker_index": f["marker_index"],
                                   "is_first_marker": f["is_first_marker"],
                                   "n_markers_in_trace": f["n_markers_in_trace"]}})
        if skipped:
            print(f"  {skipped} marker forks had no usable placebo candidate")
        probe_points(engine, tok, jobs, out, n_probes, chunk)


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

def run_branch_probes(engine, tok, branches, forks, bases, out, n_probes, max_breaks, chunk,
                      probe_every=1):
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
            if probe_every > 1 and brks:        # every n-th break, always the last
                brks = sorted(set(brks[probe_every - 1::probe_every]) | {brks[-1]})
            if len(brks) > max_breaks:          # keep an even spread, always the last
                step = len(brks) / max_breaks
                brks = sorted({brks[int(j * step)] for j in range(max_breaks)} | {brks[-1]})
            breaks_of[br["branch_id"]] = brks
            prefix = base["prompt_token_ids"] + base["output_token_ids"][:f["pos"]]
            for b in brks:
                reqs += probe_reqs(prefix + cont[:b], n_probes, f"{br['branch_id']}|{b}",
                                   suffix_ids, (br["branch_id"], b))
        by = defaultdict(list)
        for res in engine.run(reqs, stage="branch_probes"):
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
                "n_probes_per_break": n_probes, "probe_every": probe_every, **stamp(),
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
    fk.add_argument("--out", default="runs/main/forks.jsonl")

    pl = sub.add_parser("placebo")
    engine_args(pl)
    pl.add_argument("--baselines", required=True)
    pl.add_argument("--forks", required=True, help="marker forks from `probes.py forks`")
    pl.add_argument("--probes", type=int, default=4, help="same as the marker forks")
    pl.add_argument("--out", default="runs/main/placebo.jsonl")

    th = sub.add_parser("threshold")
    th.add_argument("path")
    th.add_argument("--cut", type=float, default=0.5)
    th.add_argument("--min-share", type=float, default=0.15,
                    help="warn if below/above holds less than this share. TBD")

    bp = sub.add_parser("branch")
    engine_args(bp)
    bp.add_argument("--branches", required=True)
    bp.add_argument("--forks", required=True,
                    help="fork file(s), comma-separated (marker and placebo)")
    bp.add_argument("--baselines", required=True)
    bp.add_argument("--probes", type=int, default=1, help="probe samples per break")
    bp.add_argument("--probe-every", type=int, default=1,
                    help="probe every n-th paragraph break (the last one always). "
                         "2-3 cuts probe cost a lot; first-correct is then coarser")
    bp.add_argument("--max-breaks", type=int, default=48)
    bp.add_argument("--out", default="runs/main/branch_probes.jsonl")
    args = p.parse_args()

    if args.cmd == "threshold":
        threshold_report(args.path, args.cut, args.min_share)
        return

    # Check inputs before the slow model load.
    bases = load_input(args.baselines)
    if args.cmd == "branch":
        forks = {f["fork_id"]: f for f in load_inputs(args.forks)}
        branches = load_input(args.branches)
    if args.cmd == "placebo":
        forks = load_input(args.forks)

    tok = load_tokenizer(args.model)
    engine = Engine(tok, MarkerBan(tok), args.model, args.ban_impl,
                    gpu_memory_utilization=args.gpu_mem,
                    log_path=os.path.join(os.path.dirname(args.out) or ".",
                                          "timing_log.jsonl"))
    if args.cmd == "forks":
        run_forks(engine, tok, bases, args.out, args.forks_per_trace, args.probes,
                  args.chunk)
    elif args.cmd == "placebo":
        run_placebo(engine, tok, bases, forks, args.out, args.probes, args.chunk)
    else:
        bases = {(r["unique_id"], r["seed"]): r for r in bases}
        run_branch_probes(engine, tok, branches, forks, bases, args.out, args.probes,
                          args.max_breaks, args.chunk, args.probe_every)


if __name__ == "__main__":
    main()
