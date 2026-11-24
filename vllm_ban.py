"""
vllm_ban.py - the marker ban as a vLLM (V1 engine) per-request logits processor.

Same rule as pilot.py's Hugging Face sampler (common.MarkerBan): the single marker IDs
are set to -inf at every step, and all-"m" tokens are also set to -inf right after an
"H"/"h"/"hm" token, which blocks split spellings like "H"+"mmm". Requests opt in with
SamplingParams(extra_args={"marker_ban": {...}}); requests without it are untouched, so
free and banned arms run in the same batch.

The logits-processor interface has changed between vLLM releases. This targets the
AdapterLogitsProcessor API (vLLM 0.11 and later). If it is missing, cloud_generate.py
can fall back to --ban-impl bias (logit_bias + bad_words), which covers the common
spellings but not every split. Either way the smoke test checks the decoded text for
leaks.

This module is imported by vLLM's engine process, so it must stay importable on its
own (no imports from the rest of the project).
"""
import torch

try:
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
except ImportError:          # older or newer vLLM without this API
    AdapterLogitsProcessor = None


class RequestBan:
    """Per-request callable: (output_ids, logits) -> logits."""

    def __init__(self, cfg):
        self.single = list(cfg["single"])
        self.with_seq = list(cfg["single"]) + list(cfg["next"])
        self.prefix = set(cfg["prefix"])
        self.prev0 = cfg.get("prev0")          # last prompt token, for the first step
        self._cache = {}

    def _ids(self, seq, device):
        key = (seq, str(device))
        t = self._cache.get(key)
        if t is None:
            t = torch.tensor(self.with_seq if seq else self.single,
                             dtype=torch.long, device=device)
            self._cache[key] = t
        return t

    def __call__(self, output_ids, logits):
        prev = output_ids[-1] if len(output_ids) else self.prev0
        logits[..., self._ids(prev in self.prefix, logits.device)] = float("-inf")
        return logits


if AdapterLogitsProcessor is not None:

    class MarkerBanProcessor(AdapterLogitsProcessor):
        def is_argmax_invariant(self):
            return False                   # it changes which token is most likely

        def new_req_logits_processor(self, params):
            cfg = (getattr(params, "extra_args", None) or {}).get("marker_ban")
            return RequestBan(cfg) if cfg else None
else:
    MarkerBanProcessor = None
