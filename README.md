# wait-pilot

What happens to a reasoning model's chain of thought when its "Wait"-style re-checking
markers are banned during generation? Is a given "Wait" **causal** (it triggers a re-check
that changes the trace), **stylistic** (the model re-checks anyway with other words), or
neither?

Model: DeepSeek-R1-Distill-Llama-8B. Problems: MATH-500. The design is in
`docs/Wait-Pilot Main Run Design Doc.pdf`; the analysis plan in `docs/preregistration*.md`.

Status: pipeline validated on an A100 (smoke test and threshold pilot batch: 100 problems,
287 forks). Next: exploratory branch run on the pilot forks, then preregistration v2 and
the main run.

## Layout

```
README.md, requirements.txt (local), requirements-cloud.txt (GPU pod)
docs/        design doc, preregistration v1, v2 draft
waitpilot/   the experiment code (one Python package)
tools/       standalone helpers (environment check, 4-bit model copy, response viewer)
runs/        all data and results (not in git)
  calibration/  local Hugging Face runs (pilot.py)
  smoke/        pipeline test on the pod
  pilot/        threshold pilot batch and the exploratory branch run
  main/         main run (default output location)
  reviews/      text dumps for reading traces by hand
```

| Module (`waitpilot/`) | Purpose | Runs on |
|---|---|---|
| `common.py` | settings, marker ban (incl. split spellings like "H"+"mmm"), fork detection | both |
| `grading.py` | answer grading: every box in the final section, sets vs ordered tuples, math-verify | both |
| `cloud_generate.py` | vLLM generation: baselines, branches (free / banned / placebo arms), smoke test | pod |
| `probes.py` | forced-answer probes: fork state s, placebo forks, threshold check, branch probes | pod |
| `analyze.py` | effect estimates (length primary), marker vs placebo, absorption, spread; figures | local |
| `regrade.py` | regrade any records file offline | local |
| `pilot.py` | original local experiment with Hugging Face transformers (8 GB GPU, 4-bit) | local |
| `summarize.py`, `inspect_records.py` | quick tables and readable views of `pilot.py` records | local |

Run everything **from the repository root** as a module: `python -m waitpilot.<module> ...`.

## Setup

Python 3.11 or 3.12.

**Local** (analysis, and `pilot.py` on a consumer GPU): install PyTorch with the CUDA build
for your machine (https://pytorch.org/get-started/locally/), then

```bash
python -m pip install -r requirements.txt
```

**GPU pod** (Linux): vLLM brings its own PyTorch, pinned to the version the pilot ran on.

```bash
python -m pip install -r requirements-cloud.txt
hf download deepseek-ai/DeepSeek-R1-Distill-Llama-8B      # about 16 GB, not gated
```

## Running on the pod

```bash
python -m waitpilot.cloud_generate smoke                  # must print SMOKE TEST PASSED
```

**Threshold pilot batch** (done): baselines and fork probes only.

```bash
python -m waitpilot.cloud_generate baseline --limit 100 --stratify --seeds 0 --out runs/pilot/pilot_baselines.jsonl
python -m waitpilot.probes forks --baselines runs/pilot/pilot_baselines.jsonl --out runs/pilot/pilot_forks.jsonl
python -m waitpilot.probes threshold runs/pilot/pilot_forks.jsonl
```

**Exploratory branch run on the pilot forks**: marker arms plus matched placebo words.

```bash
python -m waitpilot.probes placebo --baselines runs/pilot/pilot_baselines.jsonl --forks runs/pilot/pilot_forks.jsonl --out runs/pilot/pilot_placebo.jsonl
python -m waitpilot.cloud_generate branch --forks runs/pilot/pilot_forks.jsonl,runs/pilot/pilot_placebo.jsonl --baselines runs/pilot/pilot_baselines.jsonl --out runs/pilot/pilot_branches.jsonl
python -m waitpilot.probes branch --branches runs/pilot/pilot_branches.jsonl --forks runs/pilot/pilot_forks.jsonl,runs/pilot/pilot_placebo.jsonl --baselines runs/pilot/pilot_baselines.jsonl --out runs/pilot/pilot_branch_probes.jsonl --probe-every 3
```

**Main run** (after preregistration v2 is tagged): the same steps without `--limit`, two
seeds, outputs under `runs/main/` (the defaults).

Every step resumes where it stopped. Every command starts with a greedy ban check and stops
if the ban has no effect (`--ban-impl ids`, the default, passes the exact ban rule to vLLM as
token sequences). Every vLLM call is logged to `timing_log.jsonl` next to the output file:
stage, requests, tokens, seconds and tokens per second. Token IDs are saved with every
record, so fork prefixes are rebuilt exactly, and every record carries the git commit it was
made with (`code_version`).

## Analysis (local, no GPU)

```bash
python -m waitpilot.analyze --forks runs/pilot/pilot_forks.jsonl,runs/pilot/pilot_placebo.jsonl --branches runs/pilot/pilot_branches.jsonl --branch-probes runs/pilot/pilot_branch_probes.jsonl --out-dir runs/pilot/analysis --label exploratory
```

Writes `report.txt`, `summary.json`, `per_fork.csv` and figures. Length (total tokens) is
the primary outcome; a negative effect means the ban shortens the trace. Intervals are 95%
cluster-bootstrap intervals over problems.

## Local experiment (`pilot.py`, consumer GPU)

The original pilot: each problem is solved twice with the same seed, once normally and once
with every marker token banned. `--suppress-from paragraph` leaves the opening paragraph
unbanned so paired runs share it.

```bash
python -m waitpilot.pilot --scan-markers                 # which tokens are banned (no GPU)
python tools/save_4bit.py                                # once: 4-bit copy for an 8 GB card
python -m waitpilot.pilot --model F:/models/r1-distill-llama-8b-nf4 --levels 4,5 --limit 15 \
    --seeds 0,1 --suppress-from paragraph --out runs/calibration/calib_l45_para.jsonl
python -m waitpilot.summarize runs/calibration/calib_l45_para.jsonl --show-leaks
python -m waitpilot.inspect_records runs/calibration/calib_l45_para.jsonl > runs/reviews/review.txt
```

An output file only holds records made with the same model, token cap, `--suppress-from`
and ban version; `pilot.py` refuses to mix them. Local 4-bit runs are for checking the
pipeline: hard problems slow down sharply near 3,000 tokens on an 8 GB card.

## Grading

`grading.py` parses every `\boxed{}` in the final section. A gold comma list such as `-2,1`
is compared as a set, so `\boxed{1}, \boxed{-2}` is correct; bracketed answers such as
`(3, \frac{\pi}{2})` keep their order. Equivalence uses math-verify, OR a strict string and
number match. Older records can be regraded with `python -m waitpilot.regrade <file>`.
