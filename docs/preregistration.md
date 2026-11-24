# Preregistration v1: Wait-Pilot main run

Date: 2026-11-24
Status: v1, written before the threshold pilot batch. Items marked TBD are fixed in v2, before the main run starts. Nothing in this file was chosen after seeing any effect estimate.

## Question

When a reasoning model is about to say "Wait", how much does saying it change the outcome, and does that depend on whether its current answer is right or wrong?

Hypotheses:
- H1: Wait helps when the current answer is wrong and wastes tokens when it is right, so the average effect is near zero.
- H0: the effect is small and similar everywhere.

## Primary estimand

The slope of tau(s) on the state s, using all forks on the full 0-1 range, with final-answer accuracy as the outcome.

- tau(s) = E[Y | ban at fork, s] - E[Y | free at fork, s]
- State s = share of correct answers among forced-answer probes at the fork.

## Pre-specified secondary analyses

- Mean tau for three groups: s below 0.5, s equal to 0.5, s above 0.5, each reported with its fork count and a 95% interval. The headline contrast is below vs above.
- Other outcomes, reported separately: total tokens, tokens after the first correct answer, whether a re-check follows.
- Dose-response of tau over p_wait at the fork.
- Ban windows (one token, next sentence, rest of trace) on a subset of forks, to measure absorption by substitute phrases.

Everything else is exploratory.

## Threshold

Default 0.5. A pilot batch counts forks per group only, without looking at effects. If a group is nearly empty, the cut may move once, before v2 is tagged. Minimum share of forks per group: TBD.

## Weighting

Headline: every fork counts equally. Check: average per problem. Both are reported; disagreement is itself reported.

## Settings

- Model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B, BF16, vLLM
- Sampling: temperature 0.6, top-p 0.95, top_k 0
- Prompt suffix: "Please reason step by step, and put your final answer within \boxed{}."
- Token cap: 16,000
- Problems: MATH-500, levels 1-5, 2 seeds per problem
- K = 8 continuations per arm per fork
- Arms: free and banned. A forced-Wait arm is not run.
- Markers: Wait, Hmm (every spelling, matched by pattern), Actually, with case and leading-space token variants
- Ban window: on after the first paragraph break
- Forks per trace: TBD
- Probe samples per fork: TBD
- Truncated runs and other exclusions: TBD

## Analysis

- Cluster bootstrap over problems for the primary estimand, 95% intervals.
- Mixed-effects model with random effects for problem and fork; state and its interaction with the ban as fixed effects; shrinkage on per-fork scores.
- Variance of tau across forks compared with the variance expected from sampling noise at K = 8.
- Grading: math-verify, all boxes parsed, ordered pairs keep their order.

## Deviations

None yet. Any later change is added here with its date and reason, and the text above is not edited.
