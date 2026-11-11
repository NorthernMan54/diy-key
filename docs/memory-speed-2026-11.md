# Memory and speed: wasaga's five questions (2026-11-10)

Flynn's live instance "wasaga" (16 GB Mac) was swapping. After warm-up its main process held
about 13.1 GB, and between 13:00 and 15:30 the task queue waited up to 899 s and dropped 22 jobs.
The warm-up log (commit `81b251c`) added, per model: CLIP ViT-L-14-quickgelu FP32 on torch/MPS
+3,077 MB, SmolVLM2-2.2B MLX +4,268 MB, GFPGAN 1.4 on onnxruntime CoreML +5,284 MB, and RAM++
on CoreML +303 MB. The AI Key firmware runs on a 16 GB Jetson Orin NX, with InternVL2.5-1B in
BF16 at about 1.9 GB.

Everything below was measured on Saul (MacBook Pro, M1 Pro, 16 GB) with the `ai-gaps` venv:
onnxruntime 1.31.1, torch 2.15.0, transformers 5.22.2, mlx_vlm 0.7.12. The saul instance was
stopped. Each configuration ran in a fresh process. Memory is the change in macOS
`phys_footprint` (`_footprint_mb`, as at warm-up): "load" is right after the session or model
loads, "end" is after every run. Times are wall seconds, and "warm" is the median after the
first pass. Test inputs:
- 16 frames from 8 Driveway/doorbell clips: day, IR night, snow, people, cars, a travel trailer;
- 5 YuNet face crops from those frames: 32-237 px;
- 44 crops for CLIP: the 16 whole frames and 28 YOLO objects.

Scripts and outputs are in the session's scratchpad. They aren't committed, because the frames
are the user's camera footage.

## 1. GFPGAN: CoreML MLProgram vs CPU

| Provider (`DIYKEY_FACE_ENHANCE_PROVIDER`) | CoreML partitions | Load | Load MB | End MB | First face | Warm per face | vs CPU pixels |
|---|---|---|---|---|---|---|---|
| `cpu` | - | 0.4 s | +629 | +1,300 | 1.7 s | 1.33 s | - |
| `coreml` (default: NeuralNetwork) | 29; 492 of 557 nodes | 6.7 s | +725 | **+3,568** | 2.6 s | 1.25 s | mean 0.06, max 1 grey level; PSNR 60 dB |
| `mlprogram`, cold cache | 1; all 557 nodes | 2.6 s | +358 | +779 | 2.8 s | 0.48 s | mean 0.00, max 1; PSNR 91-138 dB |
| `mlprogram`, warm cache | 1 | 1.0 s | +634 | +1,239 | 0.7 s | 0.48 s | same |

By eye, comparing input, CPU, MLProgram, default CoreML and an ×8 difference image side by side for all five crops: the three
outputs are indistinguishable and the difference image is black. MLProgram runs the graph in
FP32, so it matches CPU almost bit for bit. The default format runs FP16 and differs by under
0.1 grey level. The default format's memory and slowness come from its 29 partitions with
CPU islands between them.

**Recommendation:** set `DIYKEY_FACE_ENHANCE_PROVIDER=mlprogram` on wasaga. The new toggle is
in `face_enhance.py`; the default is unchanged. Expect about 2.3-2.8 GB less than the default
format here. Flynn's standalone test showed +757 MB against +4,129 MB, and the same output
quality. Make it the default after it has run a day on wasaga.

## 2. RAM++: does MLProgram gain the same way?

RAM++ already uses MLProgram. `RamPlus._load` passes `ModelFormat=MLProgram` unless
`DIYKEY_RAM_PLUS_PROVIDER` isn't `coreml`, so the question's premise ("default format") doesn't
hold. All three providers are measured here anyway. Tags were compared on the 16 frames.

| Provider | CoreML nodes | Load | Load MB | End MB | Warm per frame | Tags vs CPU |
|---|---|---|---|---|---|---|
| CPU (`DIYKEY_RAM_PLUS_PROVIDER=cpu`) | - | 1.4 s | +3,044 | +3,595 | 1.29 s | - |
| CoreML MLProgram (current), cold | 992 of 1210, 75 partitions | **441 s** | +1,508 | +2,145 | 0.44 s | 18/18 frames identical; max prob diff < 0.0001 |
| CoreML MLProgram (current), ORT cache warm | same | **393 s** | +1,498 | +2,001 | 0.25 s | identical |
| CoreML default (NeuralNetwork) | 822 of 1210, 204 partitions | 36 s* | +4,665 | **+8,293** | 3.23 s | 18/18 identical; max prob diff 0.005 |

\* The first NeuralNetwork compile took about 7 min (20:58:57 → 21:06:11). The 36 s is a run
whose model CoreML had already compiled.

Findings:
- MLProgram already gives the gain: 1.6 GB less than CPU, about 6 GB less than the default
  format, and 3-5× faster, with the same tags.
- wasaga's "+303 MB" undercounts RAM++. Here it ends at about +2.0 GB. Part of the 13.1 GB is
  probably RAM++ charged to whichever model warmed up next, since CoreML compiles and maps
  lazily.
- The real cost is load time. Even with ORT's `ModelCacheDirectory` warm, the session took
  6.5 min to build. CoreML logged `resources.bin missing` / `Original module missing` for its
  e5rt bundle cache, so it recompiles for the ANE on each start. This fits the "1 min 41 s
  from CoreML's cache" note in `ucp4_client.py` on a faster Mac, and it is time the task
  queue waits after every restart.

**Recommendation:** keep MLProgram. Don't switch RAM++ to the default format, which costs
+8.3 GB at 3.2 s per frame. Investigate the compile cost separately. Options:
- `MLComputeUnits=CPUAndGPU`, which skips the ANE compile;
- a pre-converted `.mlpackage`;
- loading RAM++ in the background after the connection is up. Today it loads during
  the first tagged event.

## 3. A smaller VLM than SmolVLM2-2.2B

All runs went through `vlm_describe.py` itself: the same preprocess, prompt, 448 single tile,
greedy decoding and post-processing, on 8 Driveway/doorbell frames. Then a brand-text probe on
3 frames: "Read any company name, logo or brand text visible on the vehicles…". These clips have
no company van. The only real text is the Honda badge on the red CR-V and the travel trailer's
decal, so the probe is weak evidence.

| Model (backend) | Load MB | After 8+3 runs MB | Per summary (median; first) | Words (mean / max) | Quality on 8 frames |
|---|---|---|---|---|---|
| SmolVLM2-2.2B BF16 (mlx), **current** | +4,681 | **+5,236** | 4.5 s; 5.0 s | 9 / 13 | 8/8 usable, terse ("Red car and trailer parked in driveway.") |
| SmolVLM2-2.2B 4-bit (mlx, made locally: `mlx_vlm.convert -q --q-bits 4 --dtype bfloat16`, 6.6 bits/weight, vision not quantised) | +2,166 | +2,985 | 4.6 s; 5.0 s | 10 / 16 | 8/8 usable, about the same as BF16 |
| SmolVLM2-500M (mlx, `mlx-community/SmolVLM2-500M-Video-Instruct-mlx`) | +1,359 | +1,851 | 1.5 s; 1.7 s | 15 / 17 | ~5/8: "red truck in a parking lot" (car on a driveway), "two men in a dark room" (snowy driveway), leaks the prompt ("there are no people or animals") |
| InternVL2.5-1B 4-bit (mlx, `mlx-community/InternVL2_5-1B-4bit`) | +1,470 | +1,546 | 1.3 s; 3.1 s | 48 / **105** | ~3/8: ignores the 30-word rule, loops ("A 2-story RV… ×4"), invents years ("2018 grey RV") |
| **InternVL3.5-1B 4-bit** (mlx, `mlx-community/InternVL3_5-1B-4bit`, Apache-2.0) | +1,581 | **+1,750** | **1.25 s**; 1.4 s | 18 / 27 | ~6.5/8: the most specific ("A person is pushing a large cardboard box towards a red Honda car"); one hallucination ("holding a pizza") and one speculation ("possibly indicating disarray or neglect") |
| InternVL2.5-1B, the firmware's own weights (transformers, FP16 on MPS) | +3,956 | +4,028 | 0.98 s; 3.6 s | 16 / 20 | ~6.5/8: good sentences; calls the night car a "camper van", "red bowl" hallucination |

Brand-text probe (c3 red CR-V + trailer / c4 grey car + trailer / c8 red CR-V + trailer):

| Model | c3 | c4 | c8 |
|---|---|---|---|
| SmolVLM2-2.2B BF16 | "dr fmwaj" | None | "honda" ✓ |
| SmolVLM2-2.2B 4-bit | none | none | none |
| SmolVLM2-500M | "Ford" ✗ | "RV" | "Honda" ✓ |
| InternVL3.5-1B 4-bit | "Chevrolet" ✗ | "Ranger RV" (?) | "Honda" ✓ |
| InternVL2.5-1B 4-bit | "Hendon" ✗ | rambles | rambles |
| InternVL2.5-1B firmware | none | none | none |

Findings:
- Quantising SmolVLM2-2.2B to 4 bits saves 2.2 GB but no time on this M1 Pro. The summaries
  keep their quality, but it stopped reading text at all: 0 of 3, against 1 of 3 for BF16.
- The 1B-class models are 3.5-4.5× faster. Speed matters as much as memory for wasaga's queue.
- InternVL2.5-1B degrades badly at 4-bit MLX. The firmware avoids that by running it unquantised.
- InternVL3.5-1B at 4 bits is the best trade: 3.5 GB less than today, 3.6× faster, and summaries
  close to SmolVLM2-2.2B's, a little wordier and with one invented object in 8.
- Every small model guesses brands. On the one readable badge (Honda) the 2.2B BF16, the 500M
  and InternVL3.5 were right, but the small ones also invent makes. Naming company text from
  these models needs a check, e.g. the plate OCR path or a word list, not the VLM's word alone.
- The repo's `transformers` backend fails with the venv's transformers 5.22. The InternVL remote
  code raises `'InternVLChatModel' object has no attribute 'all_tied_weights_keys'`, and the
  `einops` dependency is missing. The row above used transformers 4.57.1 and einops from a
  scratch path. Even then it costs +4.0 GB for a 1.9 GB model, because the weights load on the
  CPU and then move to MPS, the same trap as CLIP's `.half()`.

**Recommendation:** try `DIYKEY_VLM_MODEL=mlx-community/InternVL3_5-1B-4bit` with
`DIYKEY_VLM=mlx` on saul, then wasaga. The mlx backend loads it as is, with no code change.
Expect about 3.5 GB less and about 3 s less per summary. If its occasional invented objects
show up in Protect, fall back to a local 4-bit SmolVLM2-2.2B, which saves 2.2 GB at today's
speed and quality but doesn't read text. Don't use SmolVLM2-500M or InternVL2.5-1B 4-bit.
Fix or drop the transformers backend separately (follow-up).

## 4. CLIP FP16 on MPS for Find Anything

Same 44 crops and 15 queries ("a photo of car", "red car", "a person walking with a cart",
"travel trailer", "white SUV with headlights on", …) in FP32 and FP16 on MPS. Protect's
similarity is `(cos-0.08)/0.32`.

| Measure | Result |
|---|---|
| Image vector FP16 vs FP32, cosine (min of 44) | 0.99994 |
| Text vector FP16 vs FP32, cosine (min of 15) | 0.999999 |
| Crop-query cosine change, mean / max (660 pairs) | 0.00009 / 0.0009 |
| Protect similarity change, mean / max | 0.03 / **0.28 points** |
| Pairs crossing the 50 % UI default | **0** of 660 |
| Crops whose top-3 queries change | 0 of 44 |
| FP32-stored image vs FP16 query (no re-index), max change | 0.04 points |
| Image encode on MPS, median | FP32 88 ms, FP16 74 ms |

Memory, each in a fresh process (footprint after load and 3 images):

| How it's loaded | Footprint |
|---|---|
| FP32 (today) | +2,474 to +2,752 MB |
| FP32 checkpoint then `.half()` (on CPU or MPS) | +2,363 to +3,751 MB: no saving, the FP32 pages stay |
| `DIYKEY_CLIP_FP16=1`: FP16 safetensors loaded straight to MPS (meta device, `assign=True`), then `torch.mps.empty_cache()` | **+889 MB** |

Through `clip_embed` itself, after one query and 16 frames, `torch`/`open_clip` already imported:

| `clip_embed` | Footprint | vs FP32 vectors |
|---|---|---|
| FP32 (today) | +2,832 MB | - |
| `DIYKEY_CLIP_FP16=1` | **+1,088 MB** | image cosine ≥ 0.99999, text ≥ 0.9999995, max 0.11 similarity points |

Without the `empty_cache()` after loading, the first image made about 2 GB of MPS staging
buffers resident (+2,085 MB). `_load_fp16` releases them.

**Recommendation:** FP16 is safe for Find Anything. Its scores differ from FP32 by under
0.3 similarity points. The existing FP32 embeddings in the AI Key database stay compatible
with FP16 queries, so nothing needs re-indexing. The saving only comes from loading FP16
weights directly, which is what the new opt-in `DIYKEY_CLIP_FP16=1` does. On the first start
it writes `~/.diy-key/models/clip-ViT-L-14-quickgelu-openai-fp16.safetensors` (855 MB) from
the OpenAI checkpoint; after that, a restart loads only FP16. CPU hosts stay FP32. Expect
about 1.6-1.9 GB less than today, or about 2.2 GB against wasaga's logged +3,077 MB.

## 5. Firmware engine files and the VLM settings

Searched `~/Code/aikey-fw/{2.0.4,2.2.2,2.2.8}` (extracted debs) and
`~/.diy-key/firmware/` (the three OTA tars). The engine names appear as strings in
`ui-ai-infer-agent` (`ubnt_ai_infer_agent`, 0.0.20.3 and 0.0.24.4), but only one ships in
any OTA:

| Engine | In OTA? | Size | Precision / what it is |
|---|---|---|---|
| `SmartFace_enhance_gan.engine` | No: named in infer-agent 0.0.24.4 only | - | Not shipped. It must come with the factory image or a download. |
| `SmartFace_refine.engine` | No | - | - |
| `Visual_encoder.engine` | No (`vision_encoder` / `vision_embedding` task) | - | - |
| `SmartMM_tag.engine`, `SmartMM_bbox.engine`, `SmartMM_text_kvcache.engine`, `SmartMM_text_tag_encoder.engine` | No | - | - |
| `SmartMM_text_img_encoder.engine` | Yes: 2.2.2 `ui-mm-text-img-encoder-models-0.0.2.0` (2.2.8 keeps it) | 237,139,948 B | TensorRT plan (`ftrt`). Its GEMM kernels are `sm80_xmma_gemm_f16f16…` (FP16; the INT8 kernel names are TensorRT's generic library). Its layers are a Swin backbone (`/visual_encoder/layers_0..3`, 18 blocks in stage 3) plus `/tagging_head/encoder/layer/crossattention`, i.e. a RAM-style tagger. At FP16, 237 MB fits a Swin-B-sized model (~120 M params), half of our Swin-L RAM++ (1.85 GB FP32 ONNX). |
| `SmartDetect_face_enhance.engine` (2.0.4) | Yes | 54,678,756 B | A detector "enhance" stage like `_fire/_gun/_tag_enhance`, not the GAN |
| `vlm-model/model.safetensors` (2.0.4 `ui-vlm-ai-models-0.0.2.0`; not in later OTAs) | Yes | 1,876,463,472 B | InternVL2.5-1B (`template internvl2_5`, LLM Qwen2.5-0.5B-Instruct), BF16, `force_image_size 448` |

`vlm_launch.sh` (2.0.4, 2.2.2, 2.2.8) passes no settings. It sources
`/usr/local/bin/slmenv/venv` and runs `python3 /usr/local/bin/vlmenv/app.py`. `app.py` builds
`InternVL2('/usr/local/lib/vlm-model')` with `model.py`'s defaults:
- `image_size=1280`: the long side is capped at 1280 before one 448×448 tile;
- `quantization=None`: BF16. `q4`/`q8` exist in `model.py` (bitsandbytes `load_in_4bit/8bit`) but are never selected;
- the tile is 448 (`build_transform(input_size=448)`).

`vlm_describe.py` already copies this.

**Recommendation:** nothing to borrow directly. The useful facts are that Ubiquiti's tagger is
FP16 and about half the size of ours, and that its VLM runs unquantised BF16. That supports FP16
(question 4) and, for the VLM, a smaller model rather than heavier quantisation of the 2.2B.

## Summary for wasaga

| Change | Expected saving | Risk |
|---|---|---|
| `DIYKEY_FACE_ENHANCE_PROVIDER=mlprogram` | ~2.3-4.5 GB | None seen: pixels match CPU |
| `DIYKEY_CLIP_FP16=1` | ~1.7-2 GB | ≤0.3 similarity points; first start writes an 855 MB file |
| `DIYKEY_VLM_MODEL=mlx-community/InternVL3_5-1B-4bit` | ~3.5 GB, and ~3 s less per summary | Slightly wordier; 1 invented object in 8; guesses brands |
| RAM++ | (already MLProgram) | 6.5-7 min CoreML compile per start: open item |
