# Preregistration v2: Wait-Pilot main run (DRAFT)

Date: 2026-11-24
Status: DRAFT. Not yet tagged. v1 (tag `prereg-v1`, `docs/preregistration.md`) is not edited; this file replaces it once tagged as `prereg-v2`. Items marked OPEN must be decided before tagging. Nothing here was chosen after seeing any ban-vs-free effect estimate; the pilot batch produced counts only.

## Changes from v1

- Primary estimand is now the slope of tau on s, with fork position as a covariate (v1 had no position covariate).
- Threshold 0.5 confirmed by the pilot batch. Minimum group share set to 15%.
- Forks per trace (3), probes per fork (4) and the "re-check follows" window (300 characters) are now stated; they were TBD in v1.
- Ban method fixed: exact token sequences (`--ban-impl ids`), including all "hmm" spellings. A single-m "Hm" is also blocked once the ban starts.
- Engine fixed: vLLM 0.34.0.
- Pilot batch results added below.

## Question

When a reasoning model is about to say "Wait", how much does saying it change the outcome, and does that depend on whether its current answer is right or wrong?

- H1: Wait helps when the current answer is wrong and wastes tokens when it is right, so the average effect is near zero.
- H0: the effect is small and similar everywhere.

## Primary estimand

The slope of tau(s) on the state s, using all forks on the full 0-1 range, with final-answer accuracy as the outcome, adjusting for fork position (prefix length in tokens).

- tau(s) = E[Y | ban at fork, s] - E[Y | free at fork, s]
- State s = share of correct answers among 4 forced-answer probes at the fork.

## Pre-specified secondary analyses

- Mean tau for three groups: s below 0.5, s equal to 0.5, s above 0.5, each with its fork count and a 95% interval. Headline contrast: below vs above.
- The same contrast restricted to later markers (not the first marker in a trace).
- Other outcomes, reported separately: total tokens, tokens after the first correct answer, whether a re-check follows.
- Dose-response of tau over p_wait at the fork.
- Ban windows (one token, next sentence, rest of trace) on a subset of forks, to measure absorption by substitute phrases.

Everything else is exploratory.

## Threshold

0.5. The pilot batch put 42% of forks below, 2% at and 55% above, so the rule is satisfied: below and above each hold at least 15% of forks. No change was needed.

## Weighting

Headline: every fork counts equally. Check: average per problem. Both are reported; disagreement is itself reported.

## Settings

- Model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B, BF16, vLLM 0.34.0, A100 80GB
- Sampling: temperature 0.6, top-p 0.95, top_k 0
- Prompt suffix: "Please reason step by step, and put your final answer within \boxed{}."
- Token cap: 16,000
- Problems: MATH-500, levels 1-5, 2 seeds per problem. OPEN: whether the 100 pilot problems are an exploratory set, with the confirmatory analysis on the other 400.
- K = 8 continuations per arm per fork
- Arms: free and banned. A forced-Wait arm is not run.
- Markers: Wait, Hmm (every spelling, including Hmmm), Actually, with case and leading-space token variants
- Ban method: exact token sequences passed to vLLM (`--ban-impl ids`), with a greedy ban check at start-up that stops the run if the ban has no effect
- Ban window: on after the first paragraph break
- Forks per trace: 3 (the first marker plus 2 randomly chosen later markers)
- Probes: 4 per fork. OPEN: probe density along branches (currently 1 per paragraph break); this is the main cost driver.
- "A re-check follows": a marker or substitute phrase appears within 300 characters after the fork
- Exclusions for truncated traces: OPEN
- Timing: every run writes its timing to a log file. OPEN until implemented.

## Pilot batch (counts only)

Code commit `85a8ddd`. 100 baseline traces (20 per level) and 287 forks.

- Accuracy 94%, 1% truncated, mean 3,208 tokens per trace.
- Forks: 121 below (42%), 7 at (2%), 159 above (55%). 256 of 287 have s of exactly 0 or 1.
- 72% of first-marker forks are below the cut against 26% of later ones; 103 of the 121 below-cut forks are in traces that end correct. This is the reason for the position covariate.
- Only 6 of 100 traces end wrong.

## Analysis

- Cluster bootstrap over problems for the primary estimand, 95% intervals.
- Mixed-effects model with random effects for problem and fork; state s and fork position as fixed effects, with their interactions with the ban; shrinkage on per-fork scores.
- Variance of tau across forks compared with the variance expected from sampling noise at K = 8.
- Grading: all boxes in the final section parsed; boxed sets compared order-insensitively for unbracketed comma lists, with ordered pairs keeping their order.

## Deviations

None yet. Any later change is added here with its date and reason, and the text above is not edited.
