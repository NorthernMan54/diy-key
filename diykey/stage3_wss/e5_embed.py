# SPDX-License-Identifier: AGPL-3.0-or-later
"""multilingual-e5-small text embeddings for Deep Understanding session search (384-d, unit).

Protect 7.3.70 searches deep-mode sessions by comparing a query vector with each session's
description vector, both from multilingual-e5-small. The query side is NL_PARSE
{querySentence, model: "multilingual-e5-small"}, and encodeSessionSearchQuery (M/409403)
accepts only a reply whose `model` (before any "@") is that name and whose txtEmbed has
exactly 384 values. Anything else is logged as "treating as no results". Live 2026-11-18
15:11 on the UCG Max: DIY-Key answered with CLIP ("NL_PARSE ran 'clip-ViT-L-14' but
'multilingual-e5-small' was requested"), so every session search returned
{"sessions": []}. The description side is the describe reply's descEmbedding, which fills
a vector(384) column (protocol.md §6).

e5 is trained with "query: " and "passage: " prefixes and mean pooling, so queries and
descriptions get those prefixes. The vectors are mean-pooled and normalised to unit length.

Off unless DIYKEY_E5=1 (torch + transformers, already present for the VLM). The model is
DIYKEY_E5_MODEL, default intfloat/multilingual-e5-small (~120 MB on first use). It runs on
the CPU, where a short sentence takes milliseconds. It is loaded once and used from several
threads under a lock.
"""
import logging
import os
import threading

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "multilingual-e5-small"  # what Protect asks for and checks (M/409403:30)
DIM = 384
_lock = threading.Lock()
_model = None  # None = not tried, False = unavailable, else (tokenizer, model)


def enabled():
    return os.environ.get("DIYKEY_E5", "0").strip().lower() in ("1", "true", "yes", "on")


def wants(model):
    """True when Protect's NL_PARSE asked for e5 (model may carry an "@version")."""
    return (model or "").split("@")[0] == MODEL_NAME


def _load():
    global _model
    with _lock:
        if _model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                name = os.environ.get("DIYKEY_E5_MODEL", "intfloat/multilingual-e5-small")
                tok = AutoTokenizer.from_pretrained(name)
                mdl = AutoModel.from_pretrained(name)
                mdl.eval()
                _model = (tok, mdl)
                log.info("e5 text encoder %s loaded (cpu)", name)
            except Exception as e:  # noqa: BLE001 -- session search just stays empty
                log.warning("e5 unavailable: %r", e)
                _model = False
        return _model or None


def _embed(text):
    if not enabled() or not (text or "").strip():
        return None
    loaded = _load()
    if not loaded:
        return None
    tok, mdl = loaded
    import torch
    with _lock, torch.no_grad():
        batch = tok([text], max_length=512, padding=True, truncation=True, return_tensors="pt")
        out = mdl(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    v = pooled[0].float().cpu().numpy()
    n = float(np.linalg.norm(v))
    vec = (v / n).tolist() if n > 0 else v.tolist()
    return vec if len(vec) == DIM else None


def query_embed(text):
    """384 floats for a search sentence, or None."""
    return _embed("query: " + (text or "").strip())


def passage_embed(text):
    """384 floats for a session description, or None."""
    return _embed("passage: " + (text or "").strip())
