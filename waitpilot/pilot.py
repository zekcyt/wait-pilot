"""
pilot.py - "Wait"-marker suppression pilot on MATH-500.

Each problem is solved twice with the same sampling seed:
  baseline  - normal generation
  suppress  - marker tokens ("Wait", "Hmm", "Actually", ...) banned at every step
One JSON line per (problem, condition, seed) goes to runs/calibration/pilot_records.jsonl.
Re-running the same command resumes: finished records are skipped.

Step 0 - check the marker tokens and the split-spelling rule (no GPU needed):
    python -m waitpilot.pilot --scan-markers

Local, RTX 3070 (pre-quantized 4-bit copy):
    python -m waitpilot.pilot --model F:/models/r1-distill-llama-8b-nf4 --limit 5 --max-new-tokens 3072

Local, quantize on the fly:
    python -m waitpilot.pilot --quantize 4bit --limit 5

Calibration on harder problems, opening paragraph left alone, 2 seeds:
    python -m waitpilot.pilot --model F:/models/r1-distill-llama-8b-nf4 --levels 4,5 --limit 15 \
        --seeds 0,1 --suppress-from paragraph --out runs/calibration/calib_l45_para.jsonl

Cloud, 24 GB GPU, BF16:
    python -m waitpilot.pilot --limit 500 --max-new-tokens 16000
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from waitpilot.common import (DEFAULT_MODEL, TEMPERATURE, TOP_P, MarkerBan, code_version,
                              count_markers, count_substitutes, load_problems,
                              paragraph_break_ids, split_thinking, truncation_phase,
                              user_message)
from waitpilot.grading import grade_output


def load_model(args):
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"device_map": "auto"}
    cfg = AutoConfig.from_pretrained(args.model)
    if getattr(cfg, "quantization_config", None) is not None:
        precision = "prequantized"            # e.g. the saved nf4 folder
    elif args.quantize == "4bit":
        precision = "4bit"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    elif args.quantize == "8bit":
        precision = "8bit"
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        precision = "bf16"
        kwargs["dtype"] = torch.bfloat16      # transformers < 4.56: torch_dtype=

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    return tok, model, precision


def make_sampler(ban, prompt_len, para_ids, suppress_from):
    """One logits processor that does marker masking, then temperature, then top-p,
    in that fixed order. Both conditions go through it (baseline with no masking),
    so the only difference between them is the mask.

    ban is a common.MarkerBan (None for the baseline). Besides the single marker IDs
    it blocks split spellings such as "H"+"mmm" by also masking all-"m" tokens right
    after an "H"/"h"/"hm" token.

    suppress_from="start":     mask from the first generated token.
    suppress_from="paragraph": mask only after the first blank line, so the opening
                               paragraph ("...Hmm, let me think about how to approach
                               this.") is generated freely and the paired runs share it.
    """
    import torch
    from transformers import LogitsProcessor

    class Sampler(LogitsProcessor):
        def __init__(self):
            self.ban = ban
            self.ids = torch.tensor(ban.single_ids) if ban else None
            self.ids_seq = (torch.tensor(ban.single_ids + ban.seq_next_ids)
                            if ban else None)
            self.para = torch.tensor(para_ids) if para_ids else None
            self.active = self.ids is not None and suppress_from == "start"

        def __call__(self, input_ids, scores):
            if self.ids is not None:
                if (not self.active and suppress_from == "paragraph"
                        and input_ids.shape[1] > prompt_len):
                    last = input_ids[0, -1:]
                    if bool(torch.isin(last, self.para.to(last.device)).any()):
                        self.active = True      # latches: stays on from here
                if self.active:
                    prev = (int(input_ids[0, -1]) if input_ids.shape[1] > prompt_len
                            else None)
                    ids = self.ids_seq if prev in self.ban._prefix_set else self.ids
                    scores[:, ids.to(scores.device)] = float("-inf")
            scores = scores / TEMPERATURE
            # top-p (nucleus), same algorithm as transformers' TopPLogitsWarper
            sorted_logits, sorted_idx = torch.sort(scores, descending=False, dim=-1)
            cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove_sorted = cum <= (1 - TOP_P)
            remove_sorted[..., -1:] = False    # always keep the most likely token
            remove = remove_sorted.scatter(1, sorted_idx, remove_sorted)
            return scores.masked_fill(remove, float("-inf"))

    return Sampler()


def marker_region_start(text, suppress_from):
    """Character offset where suppression starts, used to count markers in the same
    region for both conditions."""
    if suppress_from == "start":
        return 0
    i = text.find("\n\n")
    return i + 2 if i >= 0 else len(text)


def generate_one(model, tok, problem, condition, ban, para_ids, seed,
                 max_new_tokens, suppress_from):
    import torch
    from transformers import LogitsProcessorList

    inputs = tok.apply_chat_template(
        user_message(problem), add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    sampler = make_sampler(ban if condition == "suppress" else None,
                           prompt_len, para_ids, suppress_from)
    # temperature/top_p/top_k are switched off in transformers itself and applied
    # inside the sampler instead, so their order relative to the mask is guaranteed.
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True,
                      temperature=1.0, top_p=1.0, top_k=0,
                      logits_processor=LogitsProcessorList([sampler]),
                      pad_token_id=tok.pad_token_id)

    # Same seed in both conditions -> shared randomness, so a paired run usually
    # tracks the baseline until the first point where a marker was likely.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - t0

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=False)
    if tok.eos_token:
        text = text.replace(tok.eos_token, "")
    n_new = int(new_tokens.shape[0])
    # Token IDs are saved so any prefix can be rebuilt exactly (no re-tokenization).
    ids = {"prompt_token_ids": inputs["input_ids"][0].tolist(),
           "output_token_ids": new_tokens.tolist()}
    return text, n_new, elapsed, ids


# Settings that must match for records to be comparable within one output file.
LOCKED_SETTINGS = ("model", "max_new_tokens", "suppress_from", "ban_version")
# v1: single marker IDs (Wait/Hmm/Actually). v2: + punctuation/all-caps variants and the
# "H"+"mmm" sequence rule. Older files have no field and are treated as v1.
BAN_VERSION = "v2"
_LOCKED_DEFAULTS = {"suppress_from": "start", "ban_version": "v1"}


def load_done(path, settings):
    """Keys of finished records. Refuses to mix runs with different settings."""
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                for key in LOCKED_SETTINGS:
                    old = r.get(key, _LOCKED_DEFAULTS.get(key))
                    if old != settings[key]:
                        sys.exit(f"\n{path} already has records with {key}={old!r}, "
                                 f"but this run uses {key}={settings[key]!r}.\n"
                                 f"Use a different --out file for these settings.")
                done.add((r["unique_id"], r["condition"], r["seed"]))
    return done


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="none")
    p.add_argument("--limit", type=int, default=5, help="number of problems")
    p.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    p.add_argument("--conditions", default="baseline,suppress")
    p.add_argument("--max-new-tokens", type=int, default=3072)
    p.add_argument("--suppress-from", choices=["start", "paragraph"], default="start",
                   help="start: ban markers from the first token. paragraph: ban them "
                        "only after the opening paragraph, so paired runs share it.")
    p.add_argument("--levels", default="",
                   help="only these MATH difficulty levels, e.g. 4,5 (default: all)")
    p.add_argument("--out", default="runs/calibration/pilot_records.jsonl")
    p.add_argument("--scan-markers", action="store_true",
                   help="print the marker token IDs and exit (tokenizer only)")
    args = p.parse_args()

    if args.scan_markers:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        print(MarkerBan(tok).report(tok))
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    levels = [int(x) for x in args.levels.split(",")] if args.levels else None
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Check the output file before the slow model load.
    settings = {"model": args.model, "max_new_tokens": args.max_new_tokens,
                "suppress_from": args.suppress_from, "ban_version": BAN_VERSION}
    done = load_done(args.out, settings)
    problems = load_problems(args.limit, levels)

    tok, model, precision = load_model(args)
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    ban = MarkerBan(tok)
    para_ids = paragraph_break_ids(tok) if args.suppress_from == "paragraph" else []
    print(f"model={args.model} precision={precision} gpu={gpu}")
    print(f"suppressing {len(ban.single_ids)} token IDs (+ split-'Hmmm' rule) from "
          f"{args.suppress_from!r}; "
          f"max_new_tokens={args.max_new_tokens}")

    todo = [(row, s, c) for row in problems for s in seeds for c in conditions
            if (row["unique_id"], c, s) not in done]
    print(f"{len(problems)} problems x {len(seeds)} seed(s) x {len(conditions)} "
          f"condition(s): {len(todo)} to run, {len(done)} already in {args.out}\n")

    for k, (row, seed, cond) in enumerate(todo, 1):
        text, n_new, elapsed, ids = generate_one(
            model, tok, row["problem"], cond, ban, para_ids, seed,
            args.max_new_tokens, args.suppress_from)
        thinking, final = split_thinking(text)
        graded = grade_output(text, row["answer"])   # every box in the final section
        truncated = n_new >= args.max_new_tokens
        # Count markers only where suppression applies, the same region in both
        # conditions, so baseline vs suppress counts are comparable.
        start = marker_region_start(thinking, args.suppress_from)
        region = thinking[start:]
        rec = {
            "unique_id": row["unique_id"],
            "subject": row["subject"],
            "level": row["level"],
            "condition": cond,
            "seed": seed,
            "gold": row["answer"],
            "pred": graded["pred"],
            "preds_all": graded["preds_all"],
            "correct": graded["correct"],
            "grader": graded["grader"],
            "finished_thinking": "</think>" in text,
            "truncated": truncated,
            "truncation_phase": truncation_phase(text, truncated),
            "n_tokens": n_new,
            "n_thinking_chars": len(thinking),
            "suppress_from": args.suppress_from,
            "ban_version": BAN_VERSION,
            "region_start_char": start,
            "marker_counts": count_markers(region),
            "marker_counts_all": count_markers(thinking),
            "substitute_counts": count_substitutes(region),
            "seconds": round(elapsed, 1),
            "tokens_per_s": round(n_new / elapsed, 1) if elapsed else None,
            "model": args.model,
            "precision": precision,
            "gpu": gpu,
            "max_new_tokens": args.max_new_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": None,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_version": code_version(),
            "output": text,
            **ids,
        }
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        n_markers = sum(rec["marker_counts"].values())
        status = "TRUNC" if rec["truncated"] else ("ok" if rec["correct"] else "wrong")
        print(f"[{k}/{len(todo)}] {row['unique_id']:<40} {cond:<8} "
              f"{n_new:>5} tok {elapsed:>6.0f}s  markers={n_markers:<3} {status}")


if __name__ == "__main__":
    main()
