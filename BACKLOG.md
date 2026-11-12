# DIY-Key backlog

Open items for the macOS / Protect 7.3.70 work. Newest first within each section. Longer-term work is in [docs/future-backlog.md](docs/future-backlog.md).

## Next session

- **Tests for the new logic** (pure functions; good local-agent work): `nl_search.parse` (tags,
  time phrases, the `exact_match` rule), `second_verifier` (box conversion, IoU, verdicts),
  `ram_tagger.tag_crop`/`describe`/`thumbnail_tags`, `clip_embed.blend`.
- **Split `stage3_wss/ucp4_client.py`** (893 lines): framing + TLS into `ucp4_proto.py`, AI task
  handling (RAM video/image, verification, result POSTs) into `ai_tasks.py`, the nl-search loop
  into `nl_search.py`; keep the connection, setup RPCs and dispatch. No behaviour change; tests
  first. Also drop the module-level state and in-function `cv2` imports where it falls out.

## Investigate

- **Memory on 16 GB Macs** (Mac A swapping, 2026-11-10): measurements and recommendations are in
  [docs/memory-speed-2026-11.md](docs/memory-speed-2026-11.md). To try on Mac A:
  - `DIYKEY_FACE_ENHANCE_PROVIDER=mlprogram`: about 2.3-4.5 GB less, same pixels;
  - `DIYKEY_CLIP_FP16=1`: about 1.7 GB less, ≤0.3 similarity points;
  - `DIYKEY_VLM_MODEL=mlx-community/InternVL3_5-1B-4bit`: about 3.5 GB less, 3.6× faster.

  Open:
  - RAM++ MLProgram takes 6.5-7 min to compile on every start, even with ORT's cache;
  - the `transformers` VLM backend is broken on transformers 5.x (`all_tied_weights_keys`, no `einops`).

- Second-stage verification vs firmware (ui-infer-gateway second_verifier.py, 2.2.8):
  - No frame: live 2026-11-08 13:48, extraction failed ("no frame ... @4.750s", start == end,
    seek past the clip). Keeping the claimed type is right, as the firmware does; use its status
    "skipped", and fix the seek by fetching a short window around the thumbnail time.
  - No match: we return isValidDetection false at confidence 0.05 when the detector sees nothing.
    The firmware's verifier has no "nothing there" class: it keeps the detection unless top class
    prob >= 0.995 says another type.
  - Confidence: firmware reports min(det% + max(2, prob%/20), 99); we report our own.

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
- ~~**First search after a cold start hung in the UI**~~ (fixed 2026-10-29: CLIP preloads at
  start, `22f436e`; a second search took 0.3 s). Still to confirm: the UI no longer hangs, and
  `http.log` shows the `/api/detection-nls` duration.
- **`X-Token` re-sent after every restart**: `used_tokens` is per process, so a fresh start sends
  the old single-use token again. The console still accepts the connection; check whether it
  should only be sent before adoption completes.
- **Retroactive processing is one-shot**: Protect only starts it from `not_started`. Re-adopting
  resets it; decide whether a documented re-adopt is enough.

## Verify live

- ~~Find Anything returns results: exact (`car`), vector (`night`)~~ verified 2026-10-29 (night
  car event found; UI slider = min similarity, applied after the SQL).
- Still to check: `IMAGE_SEARCH`, and whether the 0.1 tag blend holds up with many rows and
  daytime crops.
- Second Stage Verification on a real `RequestAI` (`second_verifier_mlabel`) task.
- `thumbnailTags` rows for image (retroactive) tasks, not just video tasks.

## Build

- `NL_PARSE` synonyms and plurals: `people` gives no tag (not in Protect's vocabulary), so the
  search goes vector-only. Map words to vocabulary tags (`people` → `person`, `cars` → `car`), as
  the AI Key's `ui_slm_word_linker.json` does, so they get the exact-tag path.
- Session search: Protect also asks `NL_PARSE` for `multilingual-e5-small` (384-d,
  `smartDetectSessionsSearch.descEmbedding`); DIY-Key answers with CLIP, which Protect ignores.
- Face recognition in `recognizeKeyFrames` (`faceMeta`, 128-d SFace `faceEmbed`; pi-port has
  YuNet + SFace) and LPR (`vehicleMeta`, `lprAttrs`).
- `enhanceImage` (Legacy Camera Enhancement / face enhance) and `speechToText`.
- Descriptions from a real model: the AI Key's VLM weights are in the 2.0.4 firmware
  (`ui-vlm-ai-models`, `model.safetensors`); see whether its config names a public base model.
- PPE "batched AI Key classify" path in Protect's DS (`ppe.log`): what it sends to an AI Key.

## Operations

- A second instance end to end (`diykey_pg_init.sh` not yet run for real).
- 3 macOS-only errors in `tests/test_run_all.py` (MAC lookup shells out under a mocked
  `subprocess.run`).
- PostgreSQL statement logging is on for debugging (`ALTER SYSTEM RESET log_statement` to undo);
  `logs/postgres.log` is not rotated.
