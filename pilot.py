"""
pilot.py - "Wait"-marker suppression pilot on MATH-500.

Each problem is solved twice with the same sampling seed:
  baseline  - normal generation
  suppress  - marker tokens ("Wait", "Hmm", "Actually", ...) banned at every step
One JSON line per (problem, condition, seed) goes to outputs/pilot_records.jsonl.
Re-running the same command resumes: finished records are skipped.

Step 0 - check the marker tokens (no GPU needed):
    python pilot.py --scan-markers

Local, RTX 3070 (pre-quantized 4-bit copy):
    python pilot.py --model F:/models/r1-distill-llama-8b-nf4 --limit 5 --max-new-tokens 3072

Local, quantize on the fly:
    python pilot.py --quantize 4bit --limit 5

Calibration on harder problems, opening paragraph left alone, 2 seeds:
    python pilot.py --model F:/models/r1-distill-llama-8b-nf4 --levels 4,5 --limit 15 \
        --seeds 0,1 --suppress-from paragraph --out outputs/calib_l45_para.jsonl

Cloud, 24 GB GPU, BF16:
    python pilot.py --limit 500 --max-new-tokens 16384
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MARKERS = ["Wait", "wait", "Hmm", "hmm", "Actually", "actually"]
# DeepSeek's recommended instruction for math with the R1 distills (no system prompt).
PROMPT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
TEMPERATURE, TOP_P = 0.6, 0.95

# Text-level counters. These run on the decoded output, so they also catch markers
# that slip through as unusual token splits (a leak in the suppression).
MARKER_RE = re.compile(r"\b(?:[Ww]ait|[Hh]m{2,}|[Aa]ctually)\b")
# Phrases the model may switch to once "Wait" is banned. Counted, never suppressed.
SUBSTITUTE_PATTERNS = {
    "alternatively": r"\b[Aa]lternatively\b",
    "hold_on": r"\b[Hh]old on\b",
    "let_me_check": r"\b[Ll]et me (?:double[- ]check|check|verify|re-?examine|recheck)\b",
    "but_let_me": r"\b[Bb]ut let me\b",
    "on_second_thought": r"\b[Oo]n second thought\b",
    "oh": r"\bOh\b",
}


def find_marker_token_ids(tok, markers):
    """Every vocab entry that decodes to exactly a marker, ignoring surrounding
    whitespace ("Wait", " Wait", "\\nWait", ...). Exact match, so "Waiting" is kept."""
    targets = set(markers)
    found = {}
    for tid in range(len(tok)):
        text = tok.decode([tid])
        if text.strip() in targets:
            found[tid] = text
    # Markers that are NOT a single token in some common form cannot be fully
    # suppressed by banning token IDs - report them so you can decide.
    multi = []
    for m in markers:
        for variant in (m, " " + m):
            ids = tok.encode(variant, add_special_tokens=False)
            if len(ids) != 1:
                multi.append((variant, ids, [tok.decode([i]) for i in ids]))
    return found, multi


def count_markers(text):
    counts = {}
    for m in MARKER_RE.findall(text):
        key = m.lower()
        if key.startswith("hm"):
            key = "hmm"
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_substitutes(text):
    return {k: len(re.findall(p, text)) for k, p in SUBSTITUTE_PATTERNS.items()}


def last_boxed(text):
    """Contents of the last \\boxed{...} in text, with nested braces handled."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None  # unbalanced: output was cut off inside the box


def split_thinking(text):
    """(thinking, final) around </think>. final is '' if thinking never closed."""
    if "</think>" in text:
        thinking, final = text.split("</think>", 1)
        return thinking, final
    return text, ""


def normalize(ans):
    """Light normalization for string comparison of MATH answers. Deliberately
    conservative: it can mark a correct answer wrong, rarely the reverse. Raw
    answers are saved, so everything can be regraded later (e.g. with math-verify)."""
    if ans is None:
        return None
    s = ans.strip()
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\{\s*([^{}]*?)\s*\}", r"\1", s)
    for a, b in [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\:", ""), ("\\ ", ""),
                 ("^{\\circ}", ""), ("^\\circ", ""), ("\\circ", ""),
                 ("\\%", ""), ("%", ""), ("\\$", ""), ("$", ""),
                 ("\\displaystyle", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    s = re.sub(r"^[a-zA-Z]=(?!=)", "", s)                      # "x=5" -> "5"
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)        # \frac12
    s = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", s)  # \frac{1}2
    s = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", s)               # \sqrt2
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):         # 1,000 -> 1000
        s = s.replace(",", "")
    return s


def _to_number(s):
    if s is None:
        return None
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        sign = -1 if m.group(1) else 1
        den = int(m.group(3))
        return sign * int(m.group(2)) / den if den else None
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None or p == "":
        return False
    if p == g:
        return True
    pn, gn = _to_number(p), _to_number(g)
    return pn is not None and gn is not None and abs(pn - gn) < 1e-6


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


def paragraph_break_ids(tok):
    """Token IDs whose text contains a blank line (end of a paragraph)."""
    return [tid for tid in range(len(tok)) if "\n\n" in tok.decode([tid])]


def make_sampler(suppress_ids, prompt_len, para_ids, suppress_from):
    """One logits processor that does marker masking, then temperature, then top-p,
    in that fixed order. Both conditions go through it (baseline with no masking),
    so the only difference between them is the mask.

    suppress_from="start":     mask from the first generated token.
    suppress_from="paragraph": mask only after the first blank line, so the opening
                               paragraph ("...Hmm, let me think about how to approach
                               this.") is generated freely and the paired runs share it.
    """
    import torch
    from transformers import LogitsProcessor

    class Sampler(LogitsProcessor):
        def __init__(self):
            self.ids = torch.tensor(suppress_ids) if suppress_ids else None
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
                    scores[:, self.ids.to(scores.device)] = float("-inf")
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


def generate_one(model, tok, problem, condition, marker_ids, para_ids, seed,
                 max_new_tokens, suppress_from):
    import torch
    from transformers import LogitsProcessorList

    messages = [{"role": "user", "content": f"{problem}\n{PROMPT_SUFFIX}"}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    sampler = make_sampler(marker_ids if condition == "suppress" else None,
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
    return text, n_new, elapsed


def load_problems(limit, levels=None):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if levels:
        ds = ds.filter(lambda r: r["level"] in levels)
    # Fixed shuffle: --limit 5 is a subset of --limit 50, which is a subset of 500
    # (for the same --levels), so small runs can be resumed into bigger ones.
    ds = ds.shuffle(seed=0)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


# Settings that must match for records to be comparable within one output file.
LOCKED_SETTINGS = ("model", "max_new_tokens", "suppress_from")


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
                    old = r.get(key, "start" if key == "suppress_from" else None)
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
    p.add_argument("--out", default="outputs/pilot_records.jsonl")
    p.add_argument("--scan-markers", action="store_true",
                   help="print the marker token IDs and exit (tokenizer only)")
    args = p.parse_args()

    if args.scan_markers:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        found, multi = find_marker_token_ids(tok, MARKERS)
        print(f"{len(found)} marker tokens found:")
        for tid, text in sorted(found.items(), key=lambda kv: kv[1].strip()):
            print(f"  {tid:>7}  {text!r}")
        if multi:
            print("\nNOT single tokens (banning IDs cannot fully block these):")
            for variant, ids, pieces in multi:
                print(f"  {variant!r} -> {ids} {pieces}")
        else:
            print("\nAll markers are single tokens with and without a leading space.")
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    levels = [int(x) for x in args.levels.split(",")] if args.levels else None
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Check the output file before the slow model load.
    settings = {"model": args.model, "max_new_tokens": args.max_new_tokens,
                "suppress_from": args.suppress_from}
    done = load_done(args.out, settings)
    problems = load_problems(args.limit, levels)

    tok, model, precision = load_model(args)
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    marker_ids, _ = find_marker_token_ids(tok, MARKERS)
    marker_ids = sorted(marker_ids)
    para_ids = paragraph_break_ids(tok) if args.suppress_from == "paragraph" else []
    print(f"model={args.model} precision={precision} gpu={gpu}")
    print(f"suppressing {len(marker_ids)} token IDs from {args.suppress_from!r}; "
          f"max_new_tokens={args.max_new_tokens}")

    todo = [(row, s, c) for row in problems for s in seeds for c in conditions
            if (row["unique_id"], c, s) not in done]
    print(f"{len(problems)} problems x {len(seeds)} seed(s) x {len(conditions)} "
          f"condition(s): {len(todo)} to run, {len(done)} already in {args.out}\n")

    for k, (row, seed, cond) in enumerate(todo, 1):
        text, n_new, elapsed = generate_one(
            model, tok, row["problem"], cond, marker_ids, para_ids, seed,
            args.max_new_tokens, args.suppress_from)
        thinking, final = split_thinking(text)
        pred = last_boxed(final)
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
            "pred": pred,
            "correct": is_correct(pred, row["answer"]),
            "finished_thinking": "</think>" in text,
            "truncated": n_new >= args.max_new_tokens,
            "n_tokens": n_new,
            "n_thinking_chars": len(thinking),
            "suppress_from": args.suppress_from,
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
            "output": text,
        }
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        n_markers = sum(rec["marker_counts"].values())
        status = "TRUNC" if rec["truncated"] else ("ok" if rec["correct"] else "wrong")
        print(f"[{k}/{len(todo)}] {row['unique_id']:<40} {cond:<8} "
              f"{n_new:>5} tok {elapsed:>6.0f}s  markers={n_markers:<3} {status}")


if __name__ == "__main__":
    main()
