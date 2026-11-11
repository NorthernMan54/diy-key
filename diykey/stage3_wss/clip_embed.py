# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLIP ViT-L/14 embeddings for Protect's search (768-d, unit length).

Protect searches the AI Key's ramDetections by cosine distance between a query's
text embedding (NL_PARSE txtEmbed, model "clip-ViT-L-14") and each stored
object's image embedding (thumbnailTags imgEmbed). DIY-Key makes both, so they
share one space: open_clip's ViT-L-14 with OpenAI's weights (the quickgelu variant
those weights were trained with), the public model
Protect names. Like the AI Key's text encoder, short queries get "a photo of ".

Off unless DIYKEY_CLIP=1 (torch + open_clip_torch installed; ~1.7 GB of weights
on first use). Loaded once, used from several threads under a lock.

DIYKEY_CLIP_FP16=1 (opt-in, MPS/CUDA only; CPU stays FP32) runs CLIP in FP16 from an FP16 copy
of the same weights (DIYKEY_CLIP_FP16_WEIGHTS, default ~/.diy-key/models/<FP16_FILE>; written
once from the FP32 checkpoint when missing, restart afterwards to drop that copy's memory). The
model is built on the meta device and the FP16 tensors land straight on the GPU, so no FP32 copy
stays in the process: plain .half() after an FP32 load saved nothing (+2.4 GB either way, the
allocator keeps the FP32 pages). Saul (M1 Pro) 2026-11-10, after 16 frames: +1,088 MB vs +2,832 MB
FP32 (wasaga logged +3,077 MB); 44 Driveway crops x 15 queries moved cosine by <= 0.0009, i.e. <= 0.28
points of Protect's (cos-0.08)/0.32 similarity, no result crossed 50 %, every top-3 the same.
FP32-stored image vectors still match FP16 queries (<= 0.04 points), so no re-index is needed.
docs/memory-speed-2026-11.md.
"""
import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger("diykey.clip")

MODEL_NAME = "clip-ViT-L-14"
DIM = 768
_lock = threading.Lock()
_model = None


OPEN_CLIP_NAME = "ViT-L-14-quickgelu"
FP16_FILE = "clip-ViT-L-14-quickgelu-openai-fp16.safetensors"


def enabled():
    return os.environ.get("DIYKEY_CLIP", "0") == "1"


def fp16_wanted(device):
    """DIYKEY_CLIP_FP16=1 on a GPU device. CPU FP16 is slow and partly unsupported, so a CPU
    host keeps FP32 whatever the setting."""
    return os.environ.get("DIYKEY_CLIP_FP16", "0") == "1" and device in ("mps", "cuda")


def fp16_weights_path():
    return os.path.expanduser(os.environ.get("DIYKEY_CLIP_FP16_WEIGHTS")
                              or os.path.join("~/.diy-key/models", FP16_FILE))


def _write_fp16_weights(open_clip, path):
    """One-time: the OpenAI FP32 checkpoint as FP16 safetensors (atomic rename)."""
    from safetensors.torch import save_file
    model, _, _ = open_clip.create_model_and_transforms(OPEN_CLIP_NAME, pretrained="openai",
                                                        device="cpu")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    save_file({k: v.half().contiguous() for k, v in model.state_dict().items()}, tmp)
    os.replace(tmp, path)
    log.info("CLIP: wrote FP16 weights %s; restart to free the FP32 load's memory", path)


def _load_fp16(open_clip, torch, device):
    """(model, preprocess) in FP16 with no FP32 copy left on the device."""
    from safetensors.torch import load_file
    path = fp16_weights_path()
    if not os.path.isfile(path):
        _write_fp16_weights(open_clip, path)
    with torch.device("meta"):
        model = open_clip.create_model(OPEN_CLIP_NAME, pretrained=None, device="meta")
    model = model.to_empty(device=device)
    model.load_state_dict(load_file(path, device=device), assign=True)
    # attn_mask is a non-persistent buffer (not in the weights): rebuild the causal mask.
    n = model.attn_mask.shape[0]
    model.attn_mask = torch.full((n, n), float("-inf"), device=device).triu_(1)
    if device == "mps":
        # load_file(device="mps") leaves ~2 GB of staging buffers in the MPS driver; the first
        # image made them resident (+845 -> +2,085 MB on Saul) unless they are released here.
        torch.mps.empty_cache()
    cfg = open_clip.get_pretrained_cfg(OPEN_CLIP_NAME, "openai")
    preprocess = open_clip.image_transform(
        model.visual.image_size, is_train=False, mean=cfg["mean"], std=cfg["std"],
        interpolation=cfg.get("interpolation", "bicubic"),
        resize_mode=cfg.get("resize_mode", "shortest"))
    return model, preprocess


def _load():
    """(model, preprocess, tokenizer, device), or None when unavailable."""
    global _model
    with _lock:
        if _model is None:
            try:
                import open_clip
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                fp16 = fp16_wanted(device)
                if fp16:
                    model, preprocess = _load_fp16(open_clip, torch, device)
                else:
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        OPEN_CLIP_NAME, pretrained="openai", device=device)
                model.eval()
                _model = (model, preprocess, open_clip.get_tokenizer(OPEN_CLIP_NAME), device)
                log.info("CLIP ViT-L-14 (openai) loaded on %s (%s)", device,
                         "fp16" if fp16 else "fp32")
            except Exception as e:  # noqa: BLE001 -- search just stays without embeddings
                log.warning("CLIP unavailable: %r", e)
                _model = False
        return _model or None


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 0 else v.tolist()


def image_embed(img_bgr):
    """768 floats for a BGR image (a whole frame or one object's crop), or None."""
    if not enabled() or img_bgr is None or img_bgr.size == 0:
        return None
    loaded = _load()
    if not loaded:
        return None
    model, preprocess, _, device = loaded
    import torch
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    with _lock, torch.no_grad():
        x = preprocess(pil).unsqueeze(0)
        feats = model.encode_image(x.to(device, next(model.parameters()).dtype))
    return _unit(feats.float().cpu().numpy())


def text_embed(text):
    """768 floats for a search phrase, prefixed like the AI Key's encoder; or None."""
    if not enabled():
        return None
    loaded = _load()
    if not loaded:
        return None
    model, _, tokenizer, device = loaded
    import torch
    phrase = text.strip()
    if len(phrase.split()) < 3:
        phrase = "a photo of " + phrase
    with _lock, torch.no_grad():
        feats = model.encode_text(tokenizer([phrase]).to(device))
    return _unit(feats.float().cpu().numpy())


# Zero-shot labelling (ram_tagger.not_animal_prob): each label is the mean of these prompts, as
# in CLIP's own zero-shot recipe; the second covers IR night crops, which most of ours are. Both
# have 3+ words, so text_embed adds no prefix. Text vectors are cached per label set.
LABEL_TEMPLATES = ("a photo of a {}.", "a black and white night photo of a {}.")
_label_cache = {}


def label_matrix(labels):
    """Unit text vectors, one row per label (cached), or None without CLIP."""
    key = tuple(labels)
    if key not in _label_cache:
        rows = []
        for label in labels:
            vecs = [text_embed(t.format(label)) for t in LABEL_TEMPLATES]
            if any(v is None for v in vecs):
                return None
            v = np.mean(np.asarray(vecs, np.float32), 0)
            rows.append(v / np.linalg.norm(v))
        _label_cache[key] = np.stack(rows)
    return _label_cache[key]


def label_probs(image_vec, labels):
    """{label: probability} for a unit image vector over these labels (softmax of 100 x cosine,
    CLIP's logit scale), or None without CLIP or an image vector."""
    if not image_vec:
        return None
    m = label_matrix(labels)
    if m is None:
        return None
    logits = 100.0 * (m @ np.asarray(image_vec, np.float32))
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return dict(zip(labels, (float(x) for x in p)))


def protect_similarity(object_vec, query_vec):
    """Protect's similarity in percent, as AI Key Advanced alarms score it: pgvector
    `(1 - (a <=> b) - 0.08) / 0.32`, `<=>` being cosine distance, and the rule fires when
    100 * that >= its threshold (M/718745_aiprocessors.js:42-48). Search maps the same way."""
    a = np.asarray(object_vec, dtype=np.float64)
    b = np.asarray(query_vec, dtype=np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return 100 * (cos - 0.08) / 0.32


def blend_weight():
    """DIYKEY_CLIP_TAG_BLEND, default 0.1; an unparseable value means the default."""
    try:
        return float(os.environ.get("DIYKEY_CLIP_TAG_BLEND", "0.1"))
    except ValueError:
        log.warning("DIYKEY_CLIP_TAG_BLEND=%r is not a number; using 0.1",
                    os.environ.get("DIYKEY_CLIP_TAG_BLEND"))
        return 0.1


def blend(image_vec, tags, weight=None):
    """The stored object embedding: the image embedding nudged toward the text of its own tags
    ("a photo of car, night"), unit length. Stock CLIP puts small, blurred, IR-washed camera crops
    only ~0.2 cosine from a matching query; Protect's similarity maps cosine 0.08..0.40 to
    0..100 % and the UI asks for 50 %, so unblended crops never show. Weight 0.1
    (DIYKEY_CLIP_TAG_BLEND) lifted "night"/"car" on a night car crop to 51-68 % while
    "person"/"dog"/"bicycle" stayed at 37-41 %.

    This one vector is both the search row and the alarm input: ram_tagger.thumbnail_tags sends
    it as thumbnailTags[].imgEmbed, Protect keeps it as the object's embedding and passes that
    same vector to aiprocessor.matchRules (M/12387_aiprocessors.js:213-231), which scores it
    with protect_similarity against the rule's stored txtEmbed. So the blend moves alarm scores
    exactly as it moves search ranking: queries naming the crop's own tags gain ~15-30 points
    (an alarm at the UI's default threshold can fire at all), while queries the tags don't name
    stay put -- but a wrong tag (detector calls a van "car") now pulls the "car" score up too.
    Higher weights trade alarm precision for recall; 0 gives the raw CLIP calibration, under
    which a crop rarely clears ~40 %. No evidence yet for a better default than 0.1."""
    if not image_vec or not tags:
        return image_vec
    weight = blend_weight() if weight is None else float(weight)
    if weight <= 0:
        return image_vec
    text_vec = text_embed("a photo of " + ", ".join(tags))
    if not text_vec:
        return image_vec
    return _unit(np.asarray(image_vec, dtype=np.float32) + weight * np.asarray(text_vec, dtype=np.float32))
