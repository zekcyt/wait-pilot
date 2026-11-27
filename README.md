| Location     | Description                     | Link                                                                                             |   |   |
|--------------|---------------------------------|--------------------------------------------------------------------------------------------------|---|---|
| GitHub       | Code, Data, Preregs, decisions | https://github.com/zekcyt/wait-pilot/edit/main/README.md                                         |   |   |
| Google Docs  | Paper draft                     | https://docs.google.com/document/d/11Jt1lnb-IgHGP_uFhjyfHVoC4j_dLsopbF9doPSqDCg/edit?usp=sharing |   |   |
| Google Drive | Organization of Docs            | https://drive.google.com/drive/folders/1daVu0iCOdzMr6zL2U1MVWc_WfYgM4b0L?usp=sharing             |   |   |

# wait-pilot

A pilot study of what happens to a reasoning model's chain of thought when its "Wait"-style
re-checking markers are banned during generation.

**Question:** reasoning models like DeepSeek-R1-Distill-Llama-8B often interrupt themselves
with "Wait", "Hmm" or "Actually" and re-check their work. If those tokens are suppressed at
decoding time, does the model stop re-checking, re-check with different words, or recover
from errors less often?

Status: local pipeline built and calibrated on a consumer GPU. The main run (full precision,
rented GPU, long token budget) is still to do.

## How it works

Each MATH-500 problem is solved under two conditions with the same random seed:

| Condition | What happens |
|---|---|
| `baseline` | normal sampling |
| `suppress` | the logits of every marker token (`Wait`, ` wait`, `Hmm`, `Actually`, ...) are set to -inf |

The only difference between conditions is the ban. Sampling (temperature 0.6, top-p 0.95)
runs in one logits processor after the ban, so the order is fixed.

`--suppress-from paragraph` leaves the opening paragraph unbanned, so paired runs share the
model's usual opening ("Hmm, let me think about how to approach this.") and first diverge at
the first marker that would have appeared mid-reasoning.

Each run is saved as one JSON line: the raw output, the predicted and correct answers,
whether it was truncated, and marker and substitute-phrase counts.

## Files

| File | Purpose |
|---|---|
| `pilot.py` | local (Hugging Face) experiment; writes records to a `.jsonl` file |
| `common.py` | shared settings, marker ban (incl. split spellings like "H"+"mmm"), fork detection |
| `grading.py` | answer grading: every box in the final section, sets vs ordered tuples, math-verify |
| `regrade.py` | regrades any records file offline with `grading.py` |
| `cloud_generate.py` | vLLM main run: baselines, branches (free / banned arms, ban windows), smoke test |
| `probes.py` | forced-answer probes: fork state s, threshold check, probes along branches |
| `summarize.py` | accuracy, truncation, marker counts, paired comparison, leak check |
| `inspect_records.py` | readable side-by-side view of each problem for manual review |
| `smoke_test.py` | one-prompt check that the model loads and generates |
| `save_4bit.py` | saves a pre-quantized 4-bit copy of the model (low-memory local loading) |

## Setup

Python 3.11 or 3.12. Install PyTorch first, using the CUDA build for your machine
(copy the command from https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
```

Download the model once (about 16 GB, MIT licensed, no gating):

```bash
hf download deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

### Local, 8 GB GPU (4-bit)

Loading the full model and quantizing on the fly needs a lot of system memory. Save a 4-bit
copy once, then load that:

```bash
python save_4bit.py
python pilot.py --model F:/models/r1-distill-llama-8b-nf4 --limit 5
```

Edit `DST` in `save_4bit.py` to choose where the copy goes. Local runs are for checking the
pipeline. Use full precision for real numbers.

### Cloud main run (vLLM, BF16, A100 80GB)

The main run follows the design doc and `docs/preregistration.md`. On the pod:

```bash
python -m pip install -r requirements-cloud.txt
python cloud_generate.py smoke                         # must print SMOKE TEST PASSED

# threshold pilot batch: baselines + fork probes only, no branches
python cloud_generate.py baseline --limit 100 --stratify --seeds 0 --out runs/pilot_baselines.jsonl
python probes.py forks --baselines runs/pilot_baselines.jsonl --out runs/pilot_forks.jsonl
python probes.py threshold runs/pilot_forks.jsonl      # fork counts per group, no effects

# after locking the threshold and tagging preregistration v2:
python cloud_generate.py baseline --seeds 0,1 --out runs/baselines.jsonl
python probes.py forks --baselines runs/baselines.jsonl --out runs/forks.jsonl
python cloud_generate.py branch --forks runs/forks.jsonl --baselines runs/baselines.jsonl --out runs/branches.jsonl
python probes.py branch --branches runs/branches.jsonl --forks runs/forks.jsonl --baselines runs/baselines.jsonl --out runs/branch_probes.jsonl
```

Every step resumes where it stopped. Token IDs are saved with every record, so fork
prefixes are rebuilt exactly. Every command starts with a greedy ban check and stops if
the ban has no effect. The default `--ban-impl ids` passes the exact ban rule to vLLM as
token sequences; `--ban-impl words` is a fallback that misses some split spellings.

## Usage

```bash
# check which token IDs will be banned (tokenizer only, no GPU)
python pilot.py --scan-markers

# example calibration run: hard problems, 2 seeds, opening paragraph left alone
python pilot.py --model <model> --levels 4,5 --limit 15 --seeds 0,1 \
    --suppress-from paragraph --out outputs/calib_l45_para.jsonl

python summarize.py outputs/calib_l45_para.jsonl --show-leaks
python inspect_records.py outputs/calib_l45_para.jsonl > review.txt
```

| Option | Meaning |
|---|---|
| `--model` | model name or local path (a pre-quantized folder is detected automatically) |
| `--quantize` | `4bit`, `8bit` or `none` (default) |
| `--limit` | number of problems (fixed shuffle, so small runs are prefixes of larger ones) |
| `--levels` | only these MATH difficulty levels, e.g. `4,5` |
| `--seeds` | comma-separated seeds; every problem is run once per seed and condition |
| `--max-new-tokens` | generation cap; runs that hit it are marked `truncated` |
| `--suppress-from` | `start` or `paragraph` |
| `--out` | output file. Re-running the same command resumes and skips finished runs |

An output file can only hold records made with the same model, token cap and
`--suppress-from`; `pilot.py` refuses to mix them.

## Reading the results

- `acc` counts truncated runs as wrong; `acc_fin` is accuracy on runs that finished.
- Truncated runs have no final answer, so judge accuracy on finished runs.
- `substitute_counts` tracks phrases like "Alternatively" and "Let me check", to see whether
  the model re-checks with different words once the markers are banned.
- Grading (`grading.py`) parses every `\boxed{}` in the final section. A gold comma list
  such as `-2,1` is compared as a set, so `\boxed{1}, \boxed{-2}` is correct; bracketed
  answers such as `(3, \frac{\pi}{2})` keep their order. Equivalence uses math-verify.
  Records made before this grader can be regraded with `regrade.py`.

## Notes on local runs

Hard problems need more reasoning tokens than an 8 GB card can handle at a usable speed:
runs slow down sharply near 3,000 tokens, and most hard-problem runs are truncated.
Easy problems finish in 1-3 minutes. Quantitative results should come from the full-precision
run with a large token cap.

## Status of the cloud code

`cloud_generate.py` and `probes.py` were dry-run locally with a fake engine (no vLLM on
Windows). The smoke test on the pod is the real check.
