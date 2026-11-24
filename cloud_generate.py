"""
cloud_generate.py - vLLM generation for the main run (design doc, Sampling design).

Two stages, each resumable (finished records are skipped on re-run):

  baseline  One free trace per problem and seed, 16k cap. Saves prompt and output token
            IDs, so every fork prefix can be rebuilt exactly.
  branch    For each fork in forks.jsonl (from probes.py forks), K continuations per arm
            from the exact token-ID prefix:
              free         sample normally
              ban / rest   markers banned for the rest of the trace
            and on a subset of forks (--window-frac) two shorter ban windows:
              ban / token     banned for one token, then free
              ban / sentence  banned until the end of the next sentence, then free
            Comparing windows measures absorption by substitute phrases.
            The forced-"Wait" arm is not run (preregistration v1, Settings).

Typical order on the pod (see README):
    python cloud_generate.py baseline --limit 100 --stratify --seeds 0 --out runs/pilot_baselines.jsonl
    python probes.py forks --baselines runs/pilot_baselines.jsonl --out runs/pilot_forks.jsonl
    python probes.py threshold runs/pilot_forks.jsonl
    # lock the threshold, tag preregistration v2, then the main run:
    python cloud_generate.py baseline --seeds 0,1 --out runs/baselines.jsonl
    python probes.py forks --baselines runs/baselines.jsonl --out runs/forks.jsonl
    python cloud_generate.py branch --forks runs/forks.jsonl --baselines runs/baselines.jsonl --out runs/branches.jsonl
    python probes.py branch --branches runs/branches.jsonl --forks runs/forks.jsonl --out runs/branch_probes.jsonl

Smoke test (a handful of traces, checks grading, the ban and speed):
    python cloud_generate.py smoke
"""
import argparse
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from common import (DEFAULT_MODEL, MAX_NEW_TOKENS, TEMPERATURE, THINK_END, TOP_P,
                    MarkerBan, code_version, count_markers, count_substitutes, load_problems,
                    next_sentence_end, split_thinking, truncation_phase, user_message)
from grading import grade_output

# A re-check "follows" a fork if a marker or substitute phrase appears in this many
# characters of the continuation. Provisional definition; fix it in preregistration v2.
RECHECK_CHARS = 300
SENTENCE_STAGE_TOKENS = 64        # stage-1 length for the next-sentence window


def stable_seed(*parts):
    """Deterministic 31-bit seed from any key (same on every machine and run)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp():
    """Time and code version, added to every record."""
    return {"timestamp": now(), "code_version": code_version()}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, recs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


# --- engine ----------------------------------------------------------------------------

@dataclass
class Req:
    prompt_ids: list
    max_tokens: int
    seed: int
    ban: bool = False
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    logprobs: int = 0
    tag: object = None                  # caller's bookkeeping, passed through


@dataclass
class Res:
    req: Req
    token_ids: list
    finish_reason: str                  # "stop" (EOS) or "length"
    logprobs: list = field(default_factory=list)   # per step: {token_id: logprob}


class Engine:
    """Thin wrapper around vllm.LLM so the rest of the code does not depend on its API."""

    def __init__(self, tok, ban, model=DEFAULT_MODEL, ban_impl="processor",
                 max_model_len=MAX_NEW_TOKENS + 1024, gpu_memory_utilization=0.92):
        import vllm
        self.vllm_version = vllm.__version__
        self.tok, self.ban, self.ban_impl = tok, ban, ban_impl
        kwargs = dict(model=model, dtype="bfloat16", max_model_len=max_model_len,
                      gpu_memory_utilization=gpu_memory_utilization,
                      enable_prefix_caching=True, seed=0)
        if ban_impl == "processor":
            from vllm_ban import MarkerBanProcessor
            if MarkerBanProcessor is None:
                sys.exit(f"vLLM {vllm.__version__} has no AdapterLogitsProcessor. "
                         f"Use --ban-impl bias, and check the smoke test for leaks.")
            kwargs["logits_processors"] = [MarkerBanProcessor]
        self.llm = vllm.LLM(**kwargs)
        self.eos_id = tok.eos_token_id

    def _params(self, r):
        from vllm import SamplingParams
        kw = dict(temperature=r.temperature, top_p=r.top_p, top_k=-1,
                  max_tokens=r.max_tokens, seed=r.seed, skip_special_tokens=False)
        if r.logprobs:
            kw["logprobs"] = r.logprobs
        if r.ban:
            if self.ban_impl == "processor":
                kw["extra_args"] = {"marker_ban": {
                    "single": self.ban.single_ids, "prefix": self.ban.seq_prefix_ids,
                    "next": self.ban.seq_next_ids, "prev0": r.prompt_ids[-1]}}
            else:
                kw["logit_bias"] = {i: -100.0 for i in self.ban.single_ids}
                kw["bad_words"] = ["Hmmm", "hmmm", "hmm", "Hmmmm", "hmmmm"]
        return SamplingParams(**kw)

    def run(self, reqs):
        if not reqs:
            return []
        prompts = [{"prompt_token_ids": list(r.prompt_ids)} for r in reqs]
        outs = self.llm.generate(prompts, [self._params(r) for r in reqs], use_tqdm=True)
        res = []
        for r, o in zip(reqs, outs):
            c = o.outputs[0]
            lps = []
            for step in (c.logprobs or []):
                lps.append({int(k): float(v.logprob) for k, v in step.items()})
            res.append(Res(r, list(c.token_ids), c.finish_reason or "", lps))
        return res


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def load_tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def decode(tok, ids):
    text = tok.decode(ids, skip_special_tokens=False)
    return text.replace(tok.eos_token, "") if tok.eos_token else text


def prompt_ids_for(tok, problem):
    return tok.apply_chat_template(user_message(problem), add_generation_prompt=True,
                                   tokenize=True)


# --- baseline ----------------------------------------------------------------------------

def run_baseline(engine, tok, problems, seeds, out, max_new_tokens, chunk, model):
    done = {(r["unique_id"], r["seed"]) for r in load_jsonl(out)}
    todo = [(row, s) for row in problems for s in seeds if (row["unique_id"], s) not in done]
    print(f"{len(problems)} problems x {len(seeds)} seed(s): {len(todo)} to run, "
          f"{len(done)} already in {out}")
    gpu = gpu_name()
    for part in chunks(todo, chunk):
        reqs = []
        for row, s in part:
            pids = prompt_ids_for(tok, row["problem"])
            reqs.append(Req(pids, max_new_tokens, stable_seed("baseline", row["unique_id"], s),
                            tag=(row, s)))
        t0 = time.time()
        results = engine.run(reqs)
        elapsed = time.time() - t0
        recs = []
        for res in results:
            row, s = res.req.tag
            text = decode(tok, res.token_ids)
            n = len(res.token_ids)
            truncated = res.finish_reason == "length" or n >= max_new_tokens
            thinking, _ = split_thinking(text)
            region = thinking[thinking.find("\n\n") + 2:] if "\n\n" in thinking else ""
            recs.append({
                "unique_id": row["unique_id"], "subject": row["subject"],
                "level": row["level"], "seed": s, "condition": "baseline",
                "sample_seed": res.req.seed, "gold": row["answer"],
                **grade_output(text, row["answer"]),
                "finished_thinking": THINK_END in text,
                "truncated": truncated,
                "truncation_phase": truncation_phase(text, truncated),
                "finish_reason": res.finish_reason,
                "n_tokens": n,
                "marker_counts": count_markers(region),
                "marker_counts_all": count_markers(thinking),
                "substitute_counts": count_substitutes(region),
                "model": model, "precision": "bf16", "engine": f"vllm {engine.vllm_version}",
                "gpu": gpu, "max_new_tokens": max_new_tokens,
                "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": None,
                **stamp(),
                "output": text,
                "prompt_token_ids": list(res.req.prompt_ids),
                "output_token_ids": res.token_ids,
            })
        append_jsonl(out, recs)
        toks = sum(r["n_tokens"] for r in recs)
        print(f"  chunk: {len(recs)} traces, {toks} tokens in {elapsed:.0f}s "
              f"({toks / max(elapsed, 1e-9):.0f} tok/s), "
              f"acc {sum(r['correct'] for r in recs) / len(recs):.0%}, "
              f"truncated {sum(r['truncated'] for r in recs)}")


# --- branch --------------------------------------------------------------------------------

def arms_for(fork, window_frac):
    arms = [("free", "none"), ("ban", "rest")]
    # Deterministic subset of forks for the extra windows.
    if (stable_seed("window", fork["fork_id"]) % 10_000) / 10_000 < window_frac:
        arms += [("ban", "token"), ("ban", "sentence")]
    return arms


def branch_record(tok, fork, base, arm, window, k, seed, cont_ids, finish, stage1_len,
                  max_new_tokens, engine):
    text = decode(tok, cont_ids)
    total = fork["pos"] + len(cont_ids)
    truncated = finish == "length" or total >= max_new_tokens
    thinking, _ = split_thinking(text)
    # Leak check: markers inside the window that was banned.
    if arm == "ban":
        banned_text = thinking if window == "rest" else decode(tok, cont_ids[:stage1_len])
        leaks = sum(count_markers(banned_text).values())
    else:
        leaks = None
    head = text[:RECHECK_CHARS]
    return {
        "branch_id": f"{fork['fork_id']}|{arm}-{window}|k{k}",
        "fork_id": fork["fork_id"], "unique_id": fork["unique_id"], "seed": fork["seed"],
        "level": fork["level"], "arm": arm, "window": window, "k": k, "sample_seed": seed,
        "fork_pos": fork["pos"], "fork_s": fork.get("s"), "gold": base["gold"],
        **grade_output(text, base["gold"]),
        "n_tokens_total": total, "n_tokens_continuation": len(cont_ids),
        "truncated": truncated,
        "truncation_phase": truncation_phase(text, truncated),
        "finish_reason": finish,
        "ban_window_tokens": stage1_len if arm == "ban" and window != "rest" else None,
        "leaked_markers": leaks,
        "recheck_follows": bool(sum(count_markers(head).values())
                                + sum(count_substitutes(head).values())),
        "marker_counts": count_markers(thinking),
        "substitute_counts": count_substitutes(thinking),
        "engine": f"vllm {engine.vllm_version}", "ban_impl": engine.ban_impl,
        "max_new_tokens": max_new_tokens, **stamp(),
        "output": text,
        "continuation_token_ids": cont_ids,
    }


def run_branch(engine, tok, forks, bases, out, k_per_arm, window_frac, max_new_tokens,
               chunk):
    done = {r["branch_id"] for r in load_jsonl(out)}
    jobs = []   # (fork, base, arm, window, k)
    for f in forks:
        base = bases[(f["unique_id"], f["seed"])]
        for arm, window in arms_for(f, window_frac):
            for k in range(k_per_arm):
                if f"{f['fork_id']}|{arm}-{window}|k{k}" not in done:
                    jobs.append((f, base, arm, window, k))
    print(f"{len(forks)} forks: {len(jobs)} continuations to run, {len(done)} already in {out}")

    for part in chunks(jobs, chunk):
        t0 = time.time()
        one_stage, stage1 = [], []
        for f, base, arm, window, k in part:
            prefix = base["prompt_token_ids"] + base["output_token_ids"][:f["pos"]]
            budget = max_new_tokens - f["pos"]
            # Same seed across arms for continuation k (common random numbers).
            seed = stable_seed("branch", f["fork_id"], k)
            tag = (f, base, arm, window, k, seed, prefix, budget)
            if window in ("none", "rest"):
                one_stage.append(Req(prefix, budget, seed, ban=(arm == "ban"), tag=tag))
            else:
                n1 = 1 if window == "token" else min(SENTENCE_STAGE_TOKENS, budget)
                stage1.append(Req(prefix, n1, seed, ban=True, tag=tag))
        results = engine.run(one_stage + stage1)

        recs, stage2, pending = [], [], {}
        for res in results:
            f, base, arm, window, k, seed, prefix, budget = res.req.tag
            if window in ("none", "rest"):
                recs.append(branch_record(tok, f, base, arm, window, k, seed, res.token_ids,
                                          res.finish_reason, len(res.token_ids),
                                          max_new_tokens, engine))
                continue
            s1 = res.token_ids
            if window == "sentence":
                s1 = s1[:next_sentence_end(tok, s1, 0, limit=len(s1))] if s1 else s1
            ended = (res.finish_reason == "stop" and len(s1) == len(res.token_ids)
                     and s1 and s1[-1] == engine.eos_id)
            if ended or len(s1) >= budget:
                recs.append(branch_record(tok, f, base, arm, window, k, seed, s1,
                                          "stop" if ended else "length", len(s1),
                                          max_new_tokens, engine))
                continue
            key = len(stage2)
            pending[key] = (res.req.tag, s1)
            stage2.append(Req(prefix + s1, budget - len(s1), stable_seed(seed, "stage2"),
                              ban=False, tag=key))
        for res in engine.run(stage2):
            (f, base, arm, window, k, seed, prefix, budget), s1 = pending[res.req.tag]
            recs.append(branch_record(tok, f, base, arm, window, k, seed,
                                      s1 + res.token_ids, res.finish_reason, len(s1),
                                      max_new_tokens, engine))
        append_jsonl(out, recs)
        leaks = sum(1 for r in recs if r["leaked_markers"])
        toks = sum(r["n_tokens_continuation"] for r in recs)
        elapsed = time.time() - t0
        print(f"  chunk: {len(recs)} continuations, {toks} tokens in {elapsed:.0f}s "
              f"({toks / max(elapsed, 1e-9):.0f} tok/s), banned runs with a leak: {leaks}")


# --- smoke test ------------------------------------------------------------------------------

def run_smoke(engine, tok, args):
    """A few short traces: grading works, the ban leaks nothing, speed looks sane."""
    problems = load_problems(args.limit or 4, None, stratify=True)
    out_dir = args.out_dir
    base_path = os.path.join(out_dir, "smoke_baselines.jsonl")
    for p in (base_path, os.path.join(out_dir, "smoke_branches.jsonl")):
        if os.path.exists(p):
            os.remove(p)
    run_baseline(engine, tok, problems, [0], base_path, args.max_new_tokens, 64, args.model)
    bases = load_jsonl(base_path)
    print("\nbaselines:")
    for b in bases:
        print(f"  {b['unique_id']:<40} L{b['level']} {b['n_tokens']:>5} tok  "
              f"gold={b['gold']!r} pred={b['pred']!r} correct={b['correct']} "
              f"trunc={b['truncation_phase']}")

    # Ban check: from the start of each baseline's first paragraph, generate banned
    # continuations and count markers in the decoded text. Must be zero.
    from common import find_marker_positions, paragraph_break_ids
    ban, para = engine.ban, paragraph_break_ids(tok)
    end_id = tok.convert_tokens_to_ids(THINK_END)
    forks = []
    for b in bases:
        pos = find_marker_positions(tok, ban, b["output_token_ids"], para, end_id)
        if pos:
            forks.append({"fork_id": f"{b['unique_id']}|s0|t{pos[0]['pos']}",
                          "unique_id": b["unique_id"], "seed": 0, "level": b["level"],
                          "pos": pos[0]["pos"]})
    branch_path = os.path.join(out_dir, "smoke_branches.jsonl")
    run_branch(engine, tok, forks, {(b["unique_id"], 0): b for b in bases}, branch_path,
               2, 1.0, args.max_new_tokens, 256)
    brs = load_jsonl(branch_path)
    banned = [r for r in brs if r["arm"] == "ban"]
    leaked = [r for r in banned if r["leaked_markers"]]
    print(f"\nbranches: {len(brs)} ({len(banned)} banned), banned with a leak: {len(leaked)}")
    for r in leaked:
        print(f"  LEAK {r['branch_id']}: {r['marker_counts']}")
    free_markers = sum(sum(r["marker_counts"].values()) for r in brs if r["arm"] == "free")
    print(f"markers in free continuations (should be > 0): {free_markers}")
    ok = not leaked and len(bases) > 0
    print("\nSMOKE TEST " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_args(sp):
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
        sp.add_argument("--ban-impl", choices=["processor", "bias"], default="processor")
        sp.add_argument("--gpu-mem", type=float, default=0.92)
        sp.add_argument("--chunk", type=int, default=512,
                        help="requests per vLLM call (results are saved after each)")

    b = sub.add_parser("baseline")
    common_args(b)
    b.add_argument("--limit", type=int, default=0, help="number of problems (0 = all)")
    b.add_argument("--levels", default="")
    b.add_argument("--stratify", action="store_true", help="equal share per level")
    b.add_argument("--seeds", default="0,1")
    b.add_argument("--out", default="runs/baselines.jsonl")

    br = sub.add_parser("branch")
    common_args(br)
    br.add_argument("--forks", required=True)
    br.add_argument("--baselines", required=True)
    br.add_argument("--k", type=int, default=8, help="continuations per arm per fork")
    br.add_argument("--window-frac", type=float, default=0.25,
                    help="share of forks that also get the token and sentence windows")
    br.add_argument("--max-forks", type=int, default=0)
    br.add_argument("--out", default="runs/branches.jsonl")

    sm = sub.add_parser("smoke")
    common_args(sm)
    sm.add_argument("--limit", type=int, default=4)
    sm.add_argument("--out-dir", default="runs/smoke")
    args = p.parse_args()
    if args.cmd == "smoke" and args.max_new_tokens == MAX_NEW_TOKENS:
        args.max_new_tokens = 4096          # keep the smoke test short

    tok = load_tokenizer(args.model)
    ban = MarkerBan(tok)
    engine = Engine(tok, ban, args.model, args.ban_impl,
                    gpu_memory_utilization=args.gpu_mem)

    if args.cmd == "baseline":
        levels = [int(x) for x in args.levels.split(",")] if args.levels else None
        problems = load_problems(args.limit or None, levels, stratify=args.stratify)
        seeds = [int(s) for s in args.seeds.split(",")]
        run_baseline(engine, tok, problems, seeds, args.out, args.max_new_tokens,
                     args.chunk, args.model)
    elif args.cmd == "branch":
        forks = load_jsonl(args.forks)
        if args.max_forks:
            forks = forks[:args.max_forks]
        bases = {(r["unique_id"], r["seed"]): r for r in load_jsonl(args.baselines)}
        run_branch(engine, tok, forks, bases, args.out, args.k, args.window_frac,
                   args.max_new_tokens, args.chunk)
    else:
        run_smoke(engine, tok, args)


if __name__ == "__main__":
    main()
