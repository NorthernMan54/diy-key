# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-image VLM description, the AI Key's "NeXT AI Summary".

The AI Key's ui-vlm-agent (firmware 2.0.4 vlm-model/model.py, 2.2.8
vlmenv/app.py) runs InternVL2-1B on one still per event and returns a short
sentence. What it does, which this module reproduces:

  pre   crop the top 10% of the frame (Protect burns its timestamp overlay
        there), then shrink so the long side is at most 1280 px (Lanczos), then
        one 448x448 tile (no dynamic tiling) with ImageNet mean/std.
  gen   greedy (do_sample False), max_new_tokens 128, repetition_penalty 1.1,
        a "compliant image summarizer" system message and a prompt asking for
        30 words or fewer about the main subject / unusual actions, ignoring
        text, plates, brands, background and timestamp, and not guessing the
        camera angle or location.
  post  delete "The image shows/depicts/captures/features " anywhere, upper-case
        the first letter, drop repeated ", "-separated phrases (the 1B model
        loops: "a car, a car, a car") and end with a full stop.

Backends (DIYKEY_VLM, default "off" so nothing loads unless asked):
  mlx           mlx_vlm on Apple Silicon. DIYKEY_VLM_MODEL names an mlx-community
                repo, e.g. SmolVLM2 or Qwen3-VL-2B (both Apache-2.0). Avoid
                Qwen2.5-VL-3B: its Qwen Research licence is non-commercial
                (the 2.5 7B/32B/72B and the 3-series small models are Apache).
  transformers  an InternVL2/2.5-1B-style checkpoint at a local path
                (DIYKEY_VLM_MODEL), loaded with trust_remote_code and its own
                .chat(). This is how a user can run the weights from their own
                AI Key firmware (/usr/local/lib/vlm-model); Ubiquiti's fine-tune
                is not redistributable, so DIY-Key never ships or fetches it.
                OpenGVLab/InternVL2_5-1B (MIT) is the public stand-in.

Imports are lazy, so the module costs nothing when off; a missing dependency
logs once and every call returns None. One inference at a time under a module
lock: these models are hundreds of MB to GBs and not thread-safe on MPS.
Any error returns None; the caller treats that as "no summary".
"""
import json
import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger("diykey.vlm")

# Firmware values (model.py resize_image / run_inference).
CROP_TOP_FRAC = 0.10
MAX_SIDE = 1280
TILE = 448
MAX_NEW_TOKENS = 128
REPETITION_PENALTY = 1.1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Paraphrase of the firmware prompt's intent, same rules in the same order.
SYSTEM = "You describe images briefly and follow the instructions exactly."
PROMPT = (
    "Summarise this image in no more than 30 words. Rules:\n"
    "- Describe only the main subject or any unusual action.\n"
    "- Ignore visible text such as licence plates and brand names.\n"
    "- Do not describe the background or the timestamp.\n"
    "- Do not guess the camera angle, the location or the setting."
)

# app.py REDUNDANT_PHRASE: removed with str.replace, i.e. anywhere, case-sensitive.
_REDUNDANT = ("The image shows ", "The image depicts ",
              "The image captures ", "The image features ")

BACKENDS = ("mlx", "transformers", "off")

_lock = threading.Lock()        # one inference at a time
_load_lock = threading.Lock()
_loaded = {}                    # (backend, model) -> callable(rgb, prompt) -> str
_failed = set()                 # (backend, model) that failed to load: logged once


def preprocess(bgr, crop_top=True):
    """BGR frame -> RGB uint8 array, top 10% cropped, long side <= 1280. crop_top=False is for
    a crop that already left the frame's timestamp band out: it would lose its own top."""
    h = bgr.shape[0]
    img = bgr[int(h * CROP_TOP_FRAC):] if crop_top else bgr
    h, w = img.shape[:2]
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / float(max(w, h))
        # int() truncation matches the firmware's PIL resize arithmetic.
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def postprocess(text):
    """Firmware clean-up of the model's reply (app.py)."""
    # Replace before stripping, as the firmware does (it never strips): a bare
    # "The image shows " must vanish, not survive as "The image shows".
    text = text or ""
    for p in _REDUNDANT:
        text = text.replace(p, "")
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    # Capitalised first, deduped second (firmware order), so a loop that
    # starts the reply ("A car, a car") keeps one repeat; exact match only.
    seen, parts = set(), []
    for part in text.split(", "):
        if part not in seen:
            seen.add(part)
            parts.append(part)
    text = ", ".join(parts)
    if text and not text.endswith("."):
        text += "."
    return text


def _mlx_backend(model_id):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    from PIL import Image

    model, processor = load(model_id)
    config = load_config(model_id)

    def run(rgb, prompt, system=None, max_tokens=None):
        # Single-tile like the firmware: hand the model a 448 square-bounded
        # image rather than letting a dynamic-resolution model tile 1280 px.
        pil = Image.fromarray(rgb)
        pil.thumbnail((TILE, TILE), Image.LANCZOS)
        formatted = apply_chat_template(processor, config,
                                        (system or SYSTEM) + "\n" + prompt, num_images=1)
        out = generate(model, processor, formatted, [pil],
                       max_tokens=max_tokens or MAX_NEW_TOKENS, temperature=0.0,
                       repetition_penalty=REPETITION_PENALTY, verbose=False)
        return getattr(out, "text", out)
    return run


def _transformers_backend(model_path):
    if os.geteuid() == 0 and os.environ.get("DIYKEY_ALLOW_ROOT_MODEL_CODE") != "1":
        # The launchd daemon runs as root and the model directory is user-writable;
        # trust_remote_code would execute user code as root. Refuse unless explicitly allowed.
        raise RuntimeError("refusing trust_remote_code model code as root; "
                           "set DIYKEY_ALLOW_ROOT_MODEL_CODE=1")
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if dev == "mps" else torch.float32
    # Firmware loads bf16 + flash-attn on CUDA; neither exists on MPS.
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        use_flash_attn=False, use_safetensors=True,
        trust_remote_code=True).eval().to(dev)
    model.system_message = SYSTEM
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                        use_fast=False)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    gen = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
               repetition_penalty=REPETITION_PENALTY)

    def run(rgb, prompt, system=None, max_tokens=None):
        # One 448x448 bicubic tile, squashed (no aspect keep), as T.Resize did.
        tile = cv2.resize(rgb, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
        x = ((x - mean) / std).unsqueeze(0).to(dev, dtype)
        # Callers hold vlm_describe._lock, so swapping the system message is safe.
        model.system_message = system or SYSTEM
        with torch.no_grad():
            # InternVL's chat() wants "<image>\n" or prepends it itself.
            return model.chat(tok, x, "<image>\n" + prompt,
                              dict(gen, max_new_tokens=max_tokens or MAX_NEW_TOKENS))
    return run


def _get_backend(name, model):
    try:
        mtime = os.path.getmtime(model)
    except OSError:
        mtime = None
    key = (name, model, mtime)
    if key in _loaded:
        return _loaded[key]
    if key in _failed:
        return None
    with _load_lock:
        if key in _loaded:
            return _loaded[key]
        try:
            if not model:
                raise ValueError("DIYKEY_VLM_MODEL is not set")
            fn = (_mlx_backend if name == "mlx" else _transformers_backend)(model)
        except Exception as e:  # ImportError included: missing deps
            _failed.add(key)
            log.warning("VLM %s backend unavailable (%s): %s", name, model, e)
            return None
        log.info("VLM %s backend loaded: %s", name, model)
        _loaded[key] = fn
        return fn


def enabled():
    return os.environ.get("DIYKEY_VLM", "off").strip().lower() in ("mlx", "transformers")


def describe(bgr, backend=None, context=None, crop_top=True):
    """Describe one BGR frame; None when off, unavailable or on any error.

    backend: an injected callable(rgb, prompt) -> str (tests), otherwise chosen
    by DIYKEY_VLM with the model from DIYKEY_VLM_MODEL.
    context: extra instructions appended to PROMPT for this call only (what triggered the
    event, what to ignore). crop_top: see preprocess.
    """
    try:
        if backend is None:
            name = os.environ.get("DIYKEY_VLM", "off").strip().lower()
            if name not in ("mlx", "transformers"):
                if name != "off" and ("bad", name) not in _failed:
                    _failed.add(("bad", name))
                    log.warning("DIYKEY_VLM=%r unknown (want %s)", name, BACKENDS)
                return None
            backend = _get_backend(name, os.environ.get("DIYKEY_VLM_MODEL", ""))
            if backend is None:
                return None
        rgb = preprocess(bgr, crop_top=crop_top)
        prompt = f"{PROMPT}\n{context}" if context else PROMPT
        with _lock:
            raw = backend(rgb, prompt)
        return postprocess(raw) or None
    except Exception as e:
        log.warning("VLM describe failed: %s", e)
        return None


# Deep-mode attributes with Protect's own describe prompt (changeDescribePrompts, M/796872):
# the prompt asks for {"description": ..., "labels": ["key:value", ...]} JSON, which needs
# more room than the 128-token summary.
ATTR_MAX_NEW_TOKENS = 384


def parse_attribute_reply(text):
    """["key:value", ...] from a describe-prompt reply, or [] when it is not that JSON. Small
    VLMs wrap JSON in prose or ``` fences, so the outermost {...} is parsed."""
    if not text:
        return []
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        obj = json.loads(text[start:end + 1])
    except ValueError:
        return []
    labels = obj.get("labels") if isinstance(obj, dict) else None
    out = []
    for label in labels if isinstance(labels, list) else []:
        if isinstance(label, str):
            key, sep, val = label.strip().partition(":")
            if sep and key and val and val.strip().lower() != key.lower():
                label = f"{key.strip()}:{val.strip()}"
                if label not in out:
                    out.append(label)
    return out


def attributes(bgr, system_prompt, user_prompt, backend=None, max_tokens=None):
    """Protect key:value labels for one object crop (BGR) using Protect's describe prompt;
    [] when off, unavailable, unparseable or on any error. backend: as describe()."""
    try:
        if backend is None:
            name = os.environ.get("DIYKEY_VLM", "off").strip().lower()
            if name not in ("mlx", "transformers"):
                return []
            backend = _get_backend(name, os.environ.get("DIYKEY_VLM_MODEL", ""))
            if backend is None:
                return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with _lock:
            raw = backend(rgb, user_prompt, system=system_prompt,
                          max_tokens=max_tokens or ATTR_MAX_NEW_TOKENS)
        labels = parse_attribute_reply(getattr(raw, "text", raw))
        if not labels:
            log.info("VLM attributes: no labels in reply %r", str(raw)[:300])
        return labels
    except Exception as e:
        log.warning("VLM attributes failed: %s", e)
        return []
