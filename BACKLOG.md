# DIY-Key backlog

Open items for the macOS / Protect 7.3.70 work. Newest first within each section. Longer-term work is in [docs/future-backlog.md](docs/future-backlog.md).

## Next session

- **Finish splitting `stage3_wss/ucp4_client.py`** (~2,300 lines). Done 2026-11-11: framing into
  `ucp4_wire.py`, event-meta helpers into `task_meta.py` (re-exported, no behaviour change). Left:
  task queue / prefetch / warm-up, the VLM summary helpers, enhance and describe handlers. Tests
  patch these by name on `ucp4_client` (`_vlm_trigger`, `_submit_task`, `_task_stats`, ...), so
  move the tests' patch targets with the code. Also drop module-level state and in-function `cv2`
  imports where it falls out.
- **Tests for the remaining pure logic**: `ram_tagger.tag_crop`, `clip_embed.blend`. (`nl_search.parse`, `thumbnail_tags`, `describe`
  and `second_verifier` have tests now.)

## Investigate

- **Deep mode sends no describe requests** (Mac A, deep mode on in Protect from 2026-11-16
  21:40): in 36 h no `RequestAI` to `:7968/describe` or `:7445/generate-embeddings`, so
  `DIYKEY_DESCRIBE_ATTRS` never runs. Protect has never sent `changeDescribePrompts` either
  (no `state/describe_prompts.json`, checked 2026-11-18), so it seems not to treat this AI Key as
  deep-capable. Find in Protect's code what gates both (`getInfo` feature flag or capability,
  camera setting, event type?). Until then no attribute labels reach Protect (species, carrying,
  ageRange, ...); species still goes out as an object tag.
- **`segment-video` is ack-only**: Protect sends `RequestAI` `:7445/segment-video` {cameraId,
  timestamp, videoUrl (10 s ubv export)} with a `camera-upload` resUrl; 13 since 2026-11-16 on
  Driveway, before deep mode was on. We reply `{}` and upload nothing. Find what it is for
  (Locate? tap-to-select?) and what the upload should hold.
- **Summary subject**: the VLM describes parked cars as part of the action ("passing by a red
  car"). Try a snapshot with the triggering object outlined (magenta, thin; prompt names the
  outline), offline on saved clips first, plain vs boxed side by side.
- **Brand search** ("rogers van", live 2026-11-11): CLIP can't read logos at 224 px, so the word
  only lowers similarity (51 %, rank 439 of 1,188; "red van" ranks it 2nd). Option: OCR on the
  object's crop, blended into the stored embedding like the tag blend. Small VLMs invent car
  makes, so don't take brands from the summary alone.
- **Keep a small crop JPEG per stored object**, so a search miss can be checked after the clip
  is gone.
- **Memory on 16 GB Macs**: applied on Mac A 2026-11-10 (`DIYKEY_FACE_ENHANCE_PROVIDER=mlprogram`,
  `DIYKEY_CLIP_FP16=1`, `DIYKEY_VLM_MODEL=mlx-community/InternVL3_5-1B-4bit`): warm-up 13.1 GB ->
  5.0 GB, swap stopped growing. With `DIYKEY_TASK_WORKERS=2` (2026-11-11): median 8.3 s/event,
  p90 queue wait 2 ms (was up to 899 s), no drops. Details:
  [docs/memory-speed-2026-11.md](docs/memory-speed-2026-11.md). Open:
  - Make MLProgram GFPGAN and FP16 CLIP the macOS defaults now they've run for days.
  - RAM++ is 1.2-3 s per call and 2-3 calls per event, the biggest stage: try one frame, or FP16
    (the AI Key's tagger runs FP16).
  - RAM++ compile on restart: 6.5-8 min cold, ~90 s once CoreML has cached it (2026-11-11).
  - Summaries take 5-8 s here vs 1.25 s on Mac B; try a single 448 px frame (the firmware uses
    one 448 tile).
  - The `transformers` VLM backend is broken on transformers 5.x (`all_tied_weights_keys`, no
    `einops`).
- **Slow clip downloads** come from the console (memory/CPU, Network app, tshark), not from mp4
  vs ubv or parallel fetches (Mac B and console measurements, 2026-11-10). Prefetch hides most of
  it; nothing to fix on our side unless it gets worse.
- ~~**"Failed to get device details from Network application … 404"**~~ parked 2026-11-16
  (Saul, harmless noise). Protect's `networkApp.deviceInfo` (M/727717:34-71) GETs
  `{networkAppApi}/devices/<mac>/details` from (a) the Wi-Fi state refresh every 60 s for
  connected non-bridge devices (Network app >= 7.4.40, M/620729:18) and (b) the camera uplink
  lookup 100 s after each camera connects (M/408614:21). A 404 is logged at error level and
  returns null; the callers skip the update and keep the old `uplinkDevice` /
  `wifiConnectionState`. Seen for this DIY-Key (`fc:ec:da:00:00:01`), emulated cameras, an
  ONVIF camera and an AI Port, none of which Network knows by those MACs.
- **`X-Token` re-sent after every restart**: `used_tokens` is per process, so a fresh start sends
  the old single-use token again. The console still accepts the connection; check whether it
  should only be sent before adoption completes.
- **Retroactive processing is one-shot**: Protect only starts it from `not_started`. Re-adopting
  resets it; decide whether a documented re-adopt is enough.

## Verify live

- `NL_PARSE` plurals and the firmware word linker (`DIYKEY_WORD_LINKER`, on Mac A since
  2026-11-11): search "trees" / "pickup" and check the log line `nl-search word linker: ... words`.
- ~~**Person re-ID search**~~ working 2026-11-16 (Saul, live on the console). UI: open a
  recording, overlay menu -> ReID, pause ("Pause to start Edge AI analysis"), tap a person ->
  "Showing N ReID Results" (last 24 h). Gate (swai.js): camera `smartDetectTypes` non-empty plus
  an available `isSearchHost` aiprocessor (DIY Key - Flynn). Route `GET
  /detection-reidentify/search/:tempId` (M/909595) searches with the tapped OSNet vector,
  `minSimilarity = 100 - (20 + sensitivity/10)`. Matches are poor (box crop with background, not
  the firmware's silhouette; loose cutoff; IR crops); measuring OSNet same/different-person
  scores on stored rows is deferred.
- `IMAGE_SEARCH`, and whether the 0.1 tag blend holds up with many rows and daytime crops.
- ~~Second Stage Verification on a real `RequestAI`~~ verified 2026-11-16/17: 18 verdicts,
  14 kept with the firmware confidence, 3 weak other types kept, 1 person -> vehicle at 0.92.
- Description colour from the tracked vehicle's crop (2026-11-15): a car passing the parked red
  car on Driveway should no longer read "Red car, ...".
- `thumbnailTags` rows for image (retroactive) tasks, not just video tasks.
- ~~Find Anything returns results: exact (`car`), vector (`night`)~~ verified 2026-10-29.
- ~~First search after a cold start hung in the UI~~ fixed 2026-10-29 (CLIP preloads at start).

## Build

Gaps against a generic AI Key feature list (2026-11-12). That list is not from Protect or the
firmware: confirm each field and action from Protect's code or a live capture before building.

- ~~**Person attributes**~~ folded into describe labels 2026-11-16: the firmware's PAR engines
  (`SmartPAR_*`, `SmartMM_*`) are TensorRT (NVIDIA only); Protect's person/face describe prompts
  ask for the same attributes, answered by the VLM path above.
- **Describe labels: built 2026-11-16 (Saul), not live-checked.** Tags -> key:value labels
  (`ram_tagger.describe_labels`: gender, clothingType, upper/lowerClothingColor, accessory,
  vehicleType + color, species, carrying:package). `changeDescribePrompts` (M/796872) is stored
  (`state/describe_prompts.json`, hash echoed as `featureFlags.describeConfigHash`, cleared on
  factory reset); with `DIYKEY_DESCRIBE_ATTRS=1` deep describe runs Protect's prompt for the
  frame's main person/vehicle/animal on its crop (+15 % margin) and sends the reply's labels
  first. To verify live: does Protect send changeDescribePrompts to us (log
  "changeDescribePrompts: N prompt(s)"), how good InternVL3.5-1B's JSON is ("describe
  attributes (person): [...]" vs "no labels in reply"), and the extra seconds per describe.
  Still missing: footwear (Protect drops it); vehicleBrand without the VLM path.
- ~~**Package attribute**~~ built 2026-11-16 (Saul): `describe_labels` sends
  `carrying:package` for a person frame with a package/parcel/box tag (Protect's `carry` map
  has no package, but `carrying:` passes the sanitizer). Not live-checked in Find Anything.
- **Support bundle**: `getSupportInfo` is ack-only. Return a bundle with our logs, queue stats and
  warm-up timings, in whatever shape Protect expects.
- **`RequestTts`** is ack-only; the firmware ships `ui-tts-agent`. Find what Protect uses it for.
- ~~**Greyed-out Alarm Manager triggers** (Package, PPE Violation, Line Crossing; seen
  2026-11-13)~~ answered 2026-11-13: none of the three depends on the AI Key's flags. The
  trigger list is Protect's `automationManager` (M/390666 `requires`, scope filters
  M/754190), and a trigger is enabled once one adopted camera matches its scope:
  - Line Crossing: a smart camera with at least one line drawn (`smartDetectLines`). All our
    cameras have `hasLineCrossing=true` and no lines, so draw one.
  - Package: a camera whose own `featureFlags.smartDetectTypes` has `package` (not
    `extendedAiFeatures`). No adopted camera here has it.
  - PPE Violation: an adopted AI Key (as for AI Trigger) plus a camera with an industrial
    `spatialIntelligenceZone` (`helmet`/`vest` enforcement). The zone editor needs
    `featureFlags.hasSafetyZones` (Backyard, Wasaga Doorbell) and, by the UI code, probably an
    adopted AI Key. Not yet tried live.
- **Throughput test**: target ~1,000 detections/hour; the most seen live is ~150/hour. Replay saved
  clips as a synthetic burst to find the ceiling with 2 workers.
- **Callback reconciliation**: after a result POST, confirm Protect stored it (e.g. the
  `ramDetections` row for each `thumbnailTags` object), not just HTTP 200; log mismatches.
- Session search: Protect also asks `NL_PARSE` for `multilingual-e5-small` (384-d,
  `smartDetectSessionsSearch.descEmbedding`); DIY-Key answers with CLIP, which Protect ignores.
- `speechToText` (the firmware runs whisper `ggml-base.en` in ui-stt-agent).
- **PPE (`ppe_mlabel`) is answered wrongly** (Saul 2026-11-16, Protect 7.3.70 DS + AI Key
  2.2.8 firmware). `ucp4_client` treats `ppe_mlabel` like `second_verifier_mlabel`
  (person/vehicle/animal verdicts); the real model is helmet/vest:
  - Protect's Rust DS (`src/services/ppe/service.rs`, `ppe.log`) batches per camera, captures a
    snapshot it serves at `/internal/ppe/snapshot/:token`, and injects `RequestAI` with
    `targetUri :7788/v1/models/ppe_mlabel/inference` and `resUrl` `/internal/ppe/result/:token`
    over the AI Key connection. "PPE result: empty (AI Key still accumulating)" is a normal
    reply; violations map to `missingHelmet` / `missingVest` / `missingSafetyGlasses`
    (M/482984), alarm checkboxes "No Hard Hat" / "No Safety Vest" (M/390666).
  - AI Key 2.2.8 `ui-infer-gateway` (FastAPI on :7788, `models/ppe.py`, `config.yaml`):
    request `params.aggEvent[]` = `{eventId, cameraId, imageURL, objects:[{trackerId,
    coords:[x,y,w,h] 0-1000]}]}`; `ppe_mlabel.engine` (N,3,224,224) letterboxed, sigmoid over
    `[helmet, vest, is_full_body]`; a crop counts only if `is_full_body >= 0.28`, and an
    object is emitted after `agg_min_samples: 3` samples per (cameraId, trackerId), max
    aggregated. Reply `{model, action, inference_time_ms, result:[{eventId, cameraId,
    objects:[{trackerId, helmet, vest}]}]}` (scores 0-1; empty objects while accumulating).
  - Not live: no `ppe.log` on the console (no industrial camera, so the DS never sends).
  - To build: answer `ppe_mlabel` with that shape (needs a helmet/vest classifier, or the
    ONNX behind `ppe_mlabel.engine` if one can be found); test with a recorded request.
- ~~Face recognition, LPR, `enhanceImage`, VLM descriptions~~ built (face: YuNet + SFace; LPR;
  GFPGAN enhance; VLM summary via mlx, InternVL3.5-1B on Mac A; the firmware runs InternVL2.5-1B
  BF16).
- ~~`NL_PARSE` synonyms and plurals~~ done 2026-11-11 (aliases, plural fallback, word linker).

## Operations

- **Resource baseline, Mac A, 2026-11-16** (`wasaga` on 037486e, up 34 h, 2 task workers):
  - diy-key: 5.8 GB footprint, 55 CPU-min in 34 h (~2.7 % of a core on average, idle between
    events); 770 events (~22/h): total median 5.1 s, p90 11.3 s; queue wait median 0, p90 1 ms;
    tagging median 3.1 s; clip fetch wait median 2.2 s, p90 10.2 s (console side).
  - diy-port (ONVIF, same Mac): ~1 GB, ~100 % of a core steady: avclient 46 % plus 4 video
    ffmpeg ~12 % each (hw decode, 10 fps, JPEG/detect frames); the Go restreamer 8 %, 34 MB,
    plus 6 audio ffmpeg (AAC -> Opus) at 0.6 % each.
  - System (16 GB): load 2.5-4, 89 MB free, 5.5 GB compressed, swap 5.2 of 6.1 GB. An active
    Screen Sharing session costs ~90 % of a core (`screensharingd`) plus WindowServer ~35 %.
  - Memory is the limit for a second instance; CPU is not, unless diy-port grows.
- **Second instance for a separate console** (3 ONVIF cameras behind diy-port), on Mac A:
  ~5.3 GB with every model; trim to ~2.5-4 GB (face, face enhance and LPR off, 1 task worker,
  VLM optional). Needs its own alias, MAC, device ID, GUID and PostgreSQL (`diykey_pg_init.sh`
  not yet run for real); launchd setup needs sudo.
- PostgreSQL statement logging is off (`log_statement = none`, checked 2026-11-16);
  `logs/postgres.log` is not rotated.
- ~~3 macOS-only errors in `tests/test_run_all.py`~~ the full suite passes on macOS (420 tests,
  2026-11-11).
