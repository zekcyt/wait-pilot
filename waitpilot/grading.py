"""
grading.py - answer extraction and grading, shared by pilot.py, regrade.py and probes.py.

Rules (design doc, Measurement > Grading):
  - Only the final section (after </think>) is graded. No </think> means no answer.
  - Every \\boxed{} in the final section is parsed, not just the last one. Repeated boxes
    with the same answer count once.
  - Gold is an unbracketed comma list ("-2,1"): the boxes are compared as a set, order
    ignored, so "\\boxed{1}, \\boxed{-2}" is correct. This is the two-box fix for
    calibration runs 61-64.
  - Gold is bracketed ("(3, \\frac{\\pi}{2})", "[0, 1)"): compared as one object, so
    order is kept.
  - Single gold: correct only if the distinct boxes reduce to one equivalent answer.
    Two different boxed answers (hedging) are wrong.
  - Equivalence: math-verify when installed, OR the strict legacy string/number match.
    The legacy match is very conservative, so the OR adds almost no false positives and
    keeps answers like "40_9" or "\\text{(C)}" that math-verify may not parse.
"""
import logging
import os
import re

from waitpilot.common import split_thinking

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    HAVE_MATH_VERIFY = True
except ImportError:            # local HF runs can still grade with the legacy matcher
    HAVE_MATH_VERIFY = False

# math-verify's timeouts use signal.alarm, which does not exist on Windows.
_TIMEOUT = None if os.name == "nt" else 5
if _TIMEOUT is None:   # it warns on every call when timeouts are off
    logging.getLogger("math_verify").setLevel(logging.ERROR)
GRADER_VERSION = "allbox-v1" + ("+mv" if HAVE_MATH_VERIFY else "+legacy")


# --- extraction ----------------------------------------------------------------------

def _brace_body(text, i):
    """Contents of the {...} group opening at text[i] == '{', or None if unbalanced."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def all_boxed(text):
    """Contents of every \\boxed{...} / \\fbox{...} in text, in order. A box cut off by
    the token cap (unbalanced) is skipped."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        body = _brace_body(text, m.end() - 1)
        if body is not None:
            out.append(body)
    return out


def last_boxed(text):
    boxes = all_boxed(text)
    return boxes[-1] if boxes else None


def probe_answer(continuation):
    """Answer from a forced-answer probe. The prompt already ends with '\\boxed{', so the
    answer runs until the brace that closes it. None if it never closes."""
    body = _brace_body("{" + continuation, 0)
    return body.strip() if body is not None else None


# --- legacy strict matcher (moved from pilot.py, unchanged) ---------------------------

def normalize(ans):
    """Light normalization for string comparison of MATH answers. Conservative: it can
    mark a correct answer wrong, rarely the reverse."""
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


def legacy_equal(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None or p == "":
        return False
    if p == g:
        return True
    pn, gn = _to_number(p), _to_number(g)
    return pn is not None and gn is not None and abs(pn - gn) < 1e-6


def _mv_equal(pred, gold):
    if not HAVE_MATH_VERIFY:
        return False
    try:
        g = _mv_parse(f"${gold}$", parsing_timeout=_TIMEOUT)
        p = _mv_parse(f"${pred}$", parsing_timeout=_TIMEOUT)
        if not g or not p:
            return False
        return bool(_mv_verify(g, p, timeout_seconds=_TIMEOUT))
    except Exception:
        return False


def answers_equal(pred, gold):
    if pred is None or gold is None or not pred.strip():
        return False
    return legacy_equal(pred, gold) or _mv_equal(pred, gold)


# --- set / tuple handling ------------------------------------------------------------

_OPEN, _CLOSE = "({[", ")}]"


def split_top(s):
    """Split on commas that are not inside (), {}, []. '1,000' stays one item."""
    s = s.strip()
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        return [s]
    parts, depth, cur = [], 0, []
    i = 0
    while i < len(s):
        c = s[i]
        if s.startswith("\\{", i) or s.startswith("\\left", i):
            depth += 1
        elif s.startswith("\\}", i) or s.startswith("\\right", i):
            depth -= 1
        elif c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _strip_set_braces(s):
    s = s.strip()
    m = re.fullmatch(r"\\\{(.*)\\\}", s, flags=re.S)
    return m.group(1) if m else s


def is_bracketed(s):
    """Ordered object: tuple, interval or point. Compared whole, order kept."""
    s = s.strip().replace("\\left", "").replace("\\right", "")
    return len(s) >= 2 and s[0] in "([" and s[-1] in ")]"


def _distinct(answers):
    out = []
    for a in answers:
        if not any(answers_equal(a, b) for b in out):
            out.append(a)
    return out


def _match_sets(preds, golds):
    """True if preds and golds pair up one to one under answers_equal."""
    if len(preds) != len(golds):
        return False
    left = list(preds)
    for g in golds:
        hit = next((p for p in left if answers_equal(p, g)), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


def grade_answers(boxes, gold):
    """Grade a list of boxed answers against gold. Returns (correct, pred_string)."""
    boxes = [b.strip() for b in boxes if b and b.strip()]
    if not boxes:
        return False, None
    distinct = _distinct(boxes)
    pred_str = ", ".join(distinct)
    gold_items = split_top(_strip_set_braces(gold))
    if is_bracketed(gold) or len(gold_items) <= 1:
        return len(distinct) == 1 and answers_equal(distinct[0], gold), pred_str
    # Unbracketed list in gold: compare as a set, from one box or several.
    pred_items = []
    for b in distinct:
        pred_items.extend(split_top(_strip_set_braces(b)))
    return _match_sets(_distinct(pred_items), gold_items), pred_str


def grade_output(output, gold):
    """Grade a full model output. Returns a dict that callers merge into records."""
    _, final = split_thinking(output)
    boxes = all_boxed(final)
    correct, pred = grade_answers(boxes, gold)
    return {"pred": pred, "preds_all": boxes, "correct": correct,
            "grader": GRADER_VERSION}
