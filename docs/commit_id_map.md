# Commit ID map

On 2026-11-25 the history from `5519dc8` onward was rewritten to edit commit
messages only. No file contents changed: each new commit has exactly the same
tree as the old one. Commits up to and including `41ce935`, and the
`prereg-v1` tag, are unchanged.

Records written by this code store `code_version` as a git commit. If a record
names an old ID below, it ran on the code in the matching new commit.

| Old ID | New ID | Commit |
|---|---|---|
| `5519dc8` | `a903624` | Main-run code: vLLM branching, probes, all-box grading, fixed marker ban |
| `119f4c9` | `27f609a` | Match 16,000 token cap to preregistration; record code version |
| `40ae11f` | `be97a39` | Fix prompt IDs on new transformers; ban via token sequences for vLLM 0.34 |
| `b2b4976` | `eeca460` | Stop with a clear error when an input file is missing or empty |
| `85a8ddd` | `0e4584d` | Merge main-run-code: main-run pipeline, tested on the pod |
| `2330f90` | `eb32b10` | Placebo arm, analyze.py, timing log; reorganize into waitpilot/ package |

Full IDs:

```
5519dc8d011606c24ea515e9e93a8121bf0d49bb -> a9036246146da96284dab94528e7d88a13010f51
119f4c979756ddc371b4de25b8ac9c965e0e548b -> 27f609ad451300e1a2b06c95129db9f823c69ffd
40ae11fdc0bee6f988fbb20820d2e85ad35af2cd -> be97a3951507d63c3a8dbe12a8f498e783aebe92
b2b497694b181adaa18cb7432c56a4a65bd30868 -> eeca460adfc05acd61efc3ab4808b4d69dfb24e1
85a8ddd3407044d0a73f212a63e6fa5cecada8fa -> 0e4584d5243c4c75ec30614874c020682a782d54
2330f90174ccb6efa84d3b1dd4496b714bdbb6ec -> eb32b10a427a0e215220c000eb167c58eba2140c
```
