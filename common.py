"""
common.py - settings and helpers shared by pilot.py (Hugging Face, local) and the cloud
pipeline (cloud_generate.py, probes.py, regrade.py), so both ban and count the same markers.

Settings here are the ones locked in docs/preregistration.md. Change them only with a
dated entry in its Deviations section.
"""
import re

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# DeepSeek's recommended instruction for math with the R1 distills (no system prompt).
PROMPT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
TEMPERATURE, TOP_P = 0.6, 0.95          # top_k off
MAX_NEW_TOKENS = 16384                  # main-run token cap
THINK_END = "</think>"

# Forced-answer probe (design doc, Measurement): appended to a prefix, then a short
# answer is sampled. Fixed before the main run; changing it changes the state s.
PROBE_SUFFIX = "\n\nFinal Answer: \\boxed{"
PROBE_MAX_TOKENS = 48

# --- markers -------------------------------------------------------------------------
# Text-level pattern: Wait, Hmm in every spelling (2+ m's), Actually, any case of the
# first letter plus all caps. Used for counting and for the leak check on decoded text.
MARKER_RE = re.compile(r"\b(?:[Ww]ait|WAIT|[Hh]m{2,}|HMM+|[Aa]ctually|ACTUALLY)\b")
# A single vocab entry counts as a marker token if it is a marker, optionally with
# leading whitespace or punctuation (" Wait", ".Wait", "(wait", "\twait" ...). Exact
# word, so "Waiting" and "actualy" are kept.
_MARKER_TOKEN_RE = re.compile(r"[^\w]*(?:[Ww]ait|WAIT|[Hh]m{2,}|HMM+|[Aa]ctually|ACTUALLY)")
# Multi-token spellings: "Hmmm" is "H"+"mmm", "hmmm" is "hm"+"mm". Banning single IDs
# cannot block those, so after an "H"/"h"/"hm" token every all-"m" token is banned.
# Side effect: "Hm" (one m) is also blocked after the ban starts. It is not a marker
# and is rare in math traces, but it is a known, recorded side effect.
_H_PREFIX_RE = re.compile(r"[^\w]*(?:[Hh]|[Hh]m)")
_M_RUN_RE = re.compile(r"m+")

# Phrases the model may switch to once markers are banned. Counted, never banned.
SUBSTITUTE_PATTERNS = {
    "alternatively": r"\b[Aa]lternatively\b",
    "hold_on": r"\b[Hh]old on\b",
    "let_me_check": r"\b[Ll]et me (?:double[- ]check|check|verify|re-?examine|recheck)\b",
    "but_let_me": r"\b[Bb]ut let me\b",
    "on_second_thought": r"\b[Oo]n second thought\b",
    "oh": r"\bOh\b",
}


def marker_kind(text):
    """'wait' / 'hmm' / 'actually' for a marker string, else None."""
    m = MARKER_RE.search(text)
    if not m:
        return None
    w = m.group(0).lower()
    return "hmm" if w.startswith("hm") else w


class MarkerBan:
    """Token-level ban spec built from a tokenizer.

    single_ids: IDs always set to -inf while the ban is on.
    seq_prefix_ids / seq_next_ids: if the previous token is in seq_prefix_ids, the IDs
    in seq_next_ids are also set to -inf (blocks "H"+"mmm" and similar splits).
    """

    def __init__(self, tok):
        vocab = [tok.decode([i]) for i in range(len(tok))]
        self.single = {i: t for i, t in enumerate(vocab)
                       if _MARKER_TOKEN_RE.fullmatch(t.strip())}
        self.seq_prefix = {i: t for i, t in enumerate(vocab)
                           if _H_PREFIX_RE.fullmatch(t.strip()) and t.rstrip() == t}
        self.seq_next = {i: t for i, t in enumerate(vocab) if _M_RUN_RE.fullmatch(t)}
        self.single_ids = sorted(self.single)
        self.seq_prefix_ids = sorted(self.seq_prefix)
        self.seq_next_ids = sorted(self.seq_next)
        self._prefix_set = set(self.seq_prefix_ids)

    def banned_after(self, prev_id):
        """IDs to mask at the next step, given the previous token ID (or None)."""
        if prev_id is not None and prev_id in self._prefix_set:
            return self.single_ids + self.seq_next_ids
        return self.single_ids

    def as_dict(self):
        return {"single_ids": self.single_ids, "seq_prefix_ids": self.seq_prefix_ids,
                "seq_next_ids": self.seq_next_ids}

    def report(self, tok, markers=("Wait", "Hmm", "Hmmm", "Hmmmm", "Actually", "WAIT")):
        """Human-readable scan: banned IDs, sequence rule, and how common spellings
        tokenize (checks that every spelling is blocked by one of the two rules)."""
        lines = [f"{len(self.single)} single marker tokens banned:"]
        for tid, text in sorted(self.single.items(), key=lambda kv: kv[1].strip().lower()):
            lines.append(f"  {tid:>7}  {text!r}")
        lines.append(f"\nsequence rule: after any of {len(self.seq_prefix)} prefix tokens "
                     f"{sorted(self.seq_prefix.values())}")
        lines.append(f"  also ban {sorted(self.seq_next.values())}")
        lines.append("\nhow spellings tokenize (blocked = first banned piece):")
        for m in markers:
            for variant in (m, m.lower(), " " + m, " " + m.lower(), "\n\n" + m):
                ids = tok.encode(variant, add_special_tokens=False)
                pieces = [tok.decode([i]) for i in ids]
                lines.append(f"  {variant!r:<14} -> {pieces}  "
                             f"{'blocked' if self.blocks(ids) else 'NOT BLOCKED'}")
        return "\n".join(lines)

    def blocks(self, ids):
        prev = None
        for i in ids:
            if i in self.banned_after(prev):
                return True
            prev = i
        return False


def paragraph_break_ids(tok):
    """Token IDs whose text contains a blank line (end of a paragraph)."""
    return [tid for tid in range(len(tok)) if "\n\n" in tok.decode([tid])]


# --- forks (token positions where the model said a marker) ----------------------------

_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)|\n")


def find_marker_positions(tok, ban, output_ids, para_ids, think_end_id):
    """Positions in output_ids where a marker starts, inside the thinking and after the
    first paragraph break (the region the ban applies to in paragraph mode).

    Returns dicts: pos (index of the marker's first token), n_tok (1, or 2 for split
    spellings like "H"+"mmm"), text, kind, token_ids, paragraph_initial.
    """
    para = set(para_ids)
    end = output_ids.index(think_end_id) if think_end_id in output_ids else len(output_ids)
    first_para = next((i for i in range(end) if output_ids[i] in para), None)
    if first_para is None:
        return []
    single = set(ban.single_ids)
    prefix, nxt = set(ban.seq_prefix_ids), set(ban.seq_next_ids)
    out = []
    i = first_para + 1
    while i < end:
        tid = output_ids[i]
        n = 0
        if tid in single:
            n = 1
        elif tid in prefix and i + 1 < end and output_ids[i + 1] in nxt:
            # "H"+"mmm" is a marker, "H"+"m" ("Hm") is not
            if MARKER_RE.search(tok.decode(output_ids[i:i + 2])):
                n = 2
        if n:
            text = tok.decode(output_ids[i:i + n])
            before = tok.decode(output_ids[max(0, i - 3):i])
            out.append({
                "pos": i, "n_tok": n, "text": text, "kind": marker_kind(text),
                "token_ids": output_ids[i:i + n],
                "paragraph_initial": (before.rstrip(" \t").endswith("\n\n")
                                      or text.lstrip(" \t").startswith("\n")),
            })
            i += n
        else:
            i += 1
    return out


def next_sentence_end(tok, output_ids, start, limit=256):
    """Index just past the end of the sentence that starts at output_ids[start]
    (first '.', '!' or '?' followed by whitespace, or a newline, after the first token)."""
    stop = min(len(output_ids), start + limit)
    for k in range(start + 1, stop + 1):
        text = tok.decode(output_ids[start + 1:k]) if k > start + 1 else ""
        if _SENTENCE_END_RE.search(text):
            return k
    return stop


def count_markers(text):
    counts = {}
    for m in MARKER_RE.findall(text):
        key = marker_kind(m)
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_substitutes(text):
    return {k: len(re.findall(p, text)) for k, p in SUBSTITUTE_PATTERNS.items()}


def split_thinking(text):
    """(thinking, final) around </think>. final is '' if thinking never closed."""
    if THINK_END in text:
        thinking, final = text.split(THINK_END, 1)
        return thinking, final
    return text, ""


def truncation_phase(text, truncated):
    """None if the run finished; 'thinking' if it hit the cap before </think>;
    'answer' if it hit the cap while writing the final answer."""
    if not truncated:
        return None
    return "answer" if THINK_END in text else "thinking"


def user_message(problem):
    return [{"role": "user", "content": f"{problem}\n{PROMPT_SUFFIX}"}]


def load_problems(limit=None, levels=None, stratify=False):
    """MATH-500 in a fixed shuffle, so --limit 5 is a prefix of --limit 50.
    stratify=True takes an equal share from each level (pilot batch)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if levels:
        ds = ds.filter(lambda r: r["level"] in levels)
    ds = ds.shuffle(seed=0)
    if not limit:
        return list(ds)
    if not stratify:
        return list(ds.select(range(min(limit, len(ds)))))
    by_level = {}
    for row in ds:
        by_level.setdefault(row["level"], []).append(row)
    out, lv = [], sorted(by_level)
    i = 0
    while len(out) < limit and any(by_level.values()):
        bucket = by_level[lv[i % len(lv)]]
        if bucket:
            out.append(bucket.pop(0))
        i += 1
    return out
