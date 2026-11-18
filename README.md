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
| `pilot.py` | runs the experiment and writes records to a `.jsonl` file |
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

### Cloud, 24 GB or larger GPU (BF16)

```bash
python pilot.py --limit 500 --max-new-tokens 16384 --suppress-from paragraph
```

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
- The built-in grader is deliberately strict (string and simple-number matching) and can
  mark equivalent forms such as `\frac{\sqrt2}{2}` and `\frac{1}{\sqrt2}` as wrong. Raw
  outputs are saved so everything can be regraded later.

## Notes on local runs

Hard problems need more reasoning tokens than an 8 GB card can handle at a usable speed:
runs slow down sharply near 3,000 tokens, and most hard-problem runs are truncated.
Easy problems finish in 1-3 minutes. Quantitative results should come from the full-precision
run with a large token cap.

## Planned

- vLLM-based generation for the cloud run
- regrading with a math-equivalence checker
- force-answer probes: when the correct answer first appears in the reasoning, how long the
  model keeps checking afterwards, and whether wrong answers get corrected
