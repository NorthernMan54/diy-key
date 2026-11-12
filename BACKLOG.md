# DIY-Key backlog

Open items for the macOS / Protect 7.3.70 work. Newest first within each section. Longer-term work is in [docs/future-backlog.md](docs/future-backlog.md).

## Next session

- **Finish splitting `stage3_wss/ucp4_client.py`** (~2,300 lines). Done 2026-11-11: framing into
  `ucp4_wire.py`, event-meta helpers into `task_meta.py` (re-exported, no behaviour change). Left:
  task queue / prefetch / warm-up, the VLM summary helpers, enhance and describe handlers. Tests
  patch these by name on `ucp4_client` (`_vlm_trigger`, `_submit_task`, `_task_stats`, ...), so
  move the tests' patch targets with the code. Also drop module-level state and in-function `cv2`
  imports where it falls out.
- **Tests for the remaining pure logic**: `second_verifier` (box conversion, IoU, verdicts),
  `ram_tagger.tag_crop`/`describe`, `clip_embed.blend`. (`nl_search.parse` plurals/linker and
  `thumbnail_tags` have tests now.)

## Investigate

- **Description colour from the wrong car** (live 2026-11-11 07:44 and 10:54, Driveway): the event
  description leads with the parked red car ("Red car, road, ...") when a passing white SUV or
  van triggered it; RAM++ colours the whole frame. Take the colour from the triggering object's
  crop (its `thumbnailMeta` box), or drop it when the crop is too small.
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
- Second-stage verification vs firmware (ui-infer-gateway second_verifier.py, 2.2.8):
  - No frame: live 2026-11-08 13:48, extraction failed ("no frame ... @4.750s", start == end,
    seek past the clip). Keeping the claimed type is right, as the firmware does; use its status
    "skipped", and fix the seek by fetching a short window around the thumbnail time.
  - No match: we return isValidDetection false at confidence 0.05 when the detector sees nothing.
    The firmware's verifier has no "nothing there" class: it keeps the detection unless top class
    prob >= 0.995 says another type.
  - Confidence: firmware reports min(det% + max(2, prob%/20), 99); we report our own.
- **Slow clip downloads** come from the console (memory/CPU, Network app, tshark), not from mp4
  vs ubv or parallel fetches (Mac B and console measurements, 2026-11-10). Prefetch hides most of
  it; nothing to fix on our side unless it gets worse.
- **"Failed to get device details from Network application … 404"** (Protect `service.log`).
  Protect asks the UniFi Network application about every adopted device's MAC and logs a 404 for
  each one Network doesn't know:
  ```
  Failed to get device details from Network application for aa:bb:cc:33:44:cc: 404
  Failed to get device details from Network application for aa:bb:cc:33:44:bb: 404
  Failed to get device details from Network application for aa:bb:cc:33:44:aa: 404
  Failed to get device details from Network application for 02:00:00:00:00:01: 404
  Failed to get device details from Network application for fc:ec:da:00:00:01: 404
  Failed to get device details from Network application for fc:ec:da:00:00:02: 404
  Failed to get device details from Network application for 02:00:00:00:00:03: 404
  ```
  `fc:ec:da:00:00:01` is this DIY-Key instance; the others are emulated cameras
  (`aa:bb:cc:33:44:*`, unifi-cam-proxy), an ONVIF camera, an AI Port and another emulated camera.
  None are Network clients with these MACs, so this is probably harmless noise. To check: which
  call makes it (find the message in `service.js`), what Protect uses the details for (uplink,
  PoE port, topology?), and whether anything a device reports (e.g. `getInfo`, discovery) could
  satisfy it.
- **`X-Token` re-sent after every restart**: `used_tokens` is per process, so a fresh start sends
  the old single-use token again. The console still accepts the connection; check whether it
  should only be sent before adoption completes.
- **Retroactive processing is one-shot**: Protect only starts it from `not_started`. Re-adopting
  resets it; decide whether a documented re-adopt is enough.

## Verify live

- `NL_PARSE` plurals and the firmware word linker (`DIYKEY_WORD_LINKER`, on Mac A since
  2026-11-11): search "trees" / "pickup" and check the log line `nl-search word linker: ... words`.
- `IMAGE_SEARCH`, and whether the 0.1 tag blend holds up with many rows and daytime crops.
- Second Stage Verification on a real `RequestAI` (`second_verifier_mlabel`) task.
- `thumbnailTags` rows for image (retroactive) tasks, not just video tasks.
- ~~Find Anything returns results: exact (`car`), vector (`night`)~~ verified 2026-10-29.
- ~~First search after a cold start hung in the UI~~ fixed 2026-10-29 (CLIP preloads at start).

## Build

Gaps against a generic AI Key feature list (2026-11-12). That list is not from Protect or the
firmware: confirm each field and action from Protect's code or a live capture before building.

- **Person attributes** (clothing type/colour, bags, hats, glasses): the firmware has PAR engines
  (`SmartPAR_lower_enhance.engine`, `SmartMM_*` in ubnt_ai_infer_agent). Find which Protect
  result fields carry them; we only send RAM++ tags today.
- **Package context** (person carrying / dropping off a package): only RAM++ `package`/`box` tags
  now. Check whether Protect has a package event or attribute an AI Key fills.
- **Support bundle**: `getSupportInfo` is ack-only. Return a bundle with our logs, queue stats and
  warm-up timings, in whatever shape Protect expects.
- **`RequestTts`** is ack-only; the firmware ships `ui-tts-agent`. Find what Protect uses it for.
- **Throughput test**: target ~1,000 detections/hour; the most seen live is ~150/hour. Replay saved
  clips as a synthetic burst to find the ceiling with 2 workers.
- **Callback reconciliation**: after a result POST, confirm Protect stored it (e.g. the
  `ramDetections` row for each `thumbnailTags` object), not just HTTP 200; log mismatches.
- Session search: Protect also asks `NL_PARSE` for `multilingual-e5-small` (384-d,
  `smartDetectSessionsSearch.descEmbedding`); DIY-Key answers with CLIP, which Protect ignores.
- `speechToText` (the firmware runs whisper `ggml-base.en` in ui-stt-agent).
- PPE "batched AI Key classify" path in Protect's DS (`ppe.log`): what it sends to an AI Key.
- ~~Face recognition, LPR, `enhanceImage`, VLM descriptions~~ built (face: YuNet + SFace; LPR;
  GFPGAN enhance; VLM summary via mlx, InternVL3.5-1B on Mac A; the firmware runs InternVL2.5-1B
  BF16).
- ~~`NL_PARSE` synonyms and plurals~~ done 2026-11-11 (aliases, plural fallback, word linker).

## Operations

- **Second instance for a separate console** (3 ONVIF cameras behind diy-port), on Mac A:
  ~5.3 GB with every model; trim to ~2.5-4 GB (face, face enhance and LPR off, 1 task worker,
  VLM optional). Needs its own alias, MAC, device ID, GUID and PostgreSQL (`diykey_pg_init.sh`
  not yet run for real); launchd setup needs sudo.
- PostgreSQL statement logging is on for debugging (`ALTER SYSTEM RESET log_statement` to undo);
  `logs/postgres.log` is not rotated.
- ~~3 macOS-only errors in `tests/test_run_all.py`~~ the full suite passes on macOS (420 tests,
  2026-11-11).
