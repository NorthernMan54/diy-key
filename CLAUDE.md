# CLAUDE.md — DIY-Key

Context for new Claude Code sessions and for collaborators (hutchx86, NorthernMan54). Read this,
then `BACKLOG.md` (open work) and `docs/macos.md` (build / configure / run on macOS).

## What this is

DIY-Key makes a Linux or macOS host present itself to UniFi Protect as an **AI Key**
(`UP-AI-KEY`, sysid `0xa5f0`) and do the AI Key's work. Upstream (hutchx86) targets an Orange Pi 5
Plus / RK3588 with systemd; the **`MacOS` branch** (NorthernMan54) runs it on a Mac against
**Protect 7.3.70** (console: UCG-Max) and implements most of the AI work for real. Interoperability
only: no vendor code or binaries in the repo, no console modification.

## Working rules

- **Commit Mac-only work on the `MacOS` branch, not `main`.** `main` is the integration branch:
  it carries this work plus the Linux/RK3588 runtime (see `docs/linux.md`), so changes merged
  there must keep both targets working. Both branches are on `origin` (hutchx86's repo); ask
  before pushing or opening a PR.
- **Never commit vendor material**: firmware, extracted packages, Protect's `service.js`/UI
  bundle, `ram_tags.csv`, model weights. They live outside the repo (see "Lab" below); reference
  them by path or env var.
- New files carry `# SPDX-License-Identifier: AGPL-3.0-or-later`. Match the surrounding style:
  module docstring that explains the *why* and cites what was confirmed live.
- Every RPC/schema fact should come from Protect's code or a live capture, and say which.
- Tests: `venv/bin/python -m unittest discover -s diykey/tests`. New pure logic gets tests
  (see BACKLOG "Next session").
- Restart a running instance after code changes: `touch ~/.diy-key/<instance>/restart`.

## Layout (diykey/)

| Path | Role |
|---|---|
| `core/config.py` | Config/identity (`diykey.cfg` + `DIYKEY_*` env), state files, log redaction, firmware version pin |
| `stage1_discovery/beacon.py` | UBNT discovery UDP/10001 |
| `stage1_http/adopt_http.py` | HTTPS :8080 (`/api/info` GET+POST, `/api/adopt`) |
| `stage3_wss/ucp4_client.py` | ucp4 to the console :7442, setup RPCs, AI task dispatch, task queue, prefetch, model warm-up, nl-search loop (~2,400 lines; split planned) |
| `stage3_wss/vlm_describe.py` | One-image VLM summary ("NeXT AI Summary"): mlx or transformers backend |
| `stage3_wss/scene_memory.py` | Per-camera "always there" scene, so summaries and tags follow what changed |
| `stage3_wss/face_recog.py` | Face detection + embedding for `recognizeFaces` |
| `stage3_wss/face_enhance.py` | GFPGAN for Faces-page Enhance (`enhanceImage`); `DIYKEY_FACE_ENHANCE_PROVIDER` |
| `stage3_wss/lpr.py` | Licence-plate reads for the LPR result |
| `stage3_wss/reid_model.py` | Person re-ID vectors (OSNet-AIN) |
| `stage3_wss/system_stats.py` | Hardware stats for the `systemMetrics` RPC |
| `stage3_wss/clip_fetch.py` | Fetch task clips (as MP4) / images from the console, grab frames |
| `stage3_wss/ram_tagger.py` | COCO ONNX detector → Protect RAM tags, descriptions, per-object `thumbnailTags` |
| `stage3_wss/second_verifier.py` | Second Stage Verification (`RequestAI` → `second_verifier_mlabel`) |
| `stage3_wss/clip_embed.py` | CLIP ViT-L/14 (OpenAI, quickgelu) image/text embeddings, tag blend |
| `stage3_wss/nl_search.py` | Find Anything `NL_PARSE` / `IMAGE_SEARCH` replies |
| `stage3_wss/firmware.py` | Keep the firmware Protect offers |
| `scripts/macos_launchd.sh` | Root launchd daemon for an instance: boot start, IP alias, independent of any terminal |
| `scripts/diykey_service.sh`, `diykey_paths.sh`, `diykey_pg_init.sh`, `diykey.env.example` | macOS per-instance service (`~/.diy-key/<instance>/`) |

## Protocol facts (Protect 7.3.70; hard to rediscover)

- **Two WebSockets.** ucp4 to `:7442` (RPCs, tasks). And the **nl-search** socket a real AI Key
  opens to `wss://<console>:7443/wss/nl-search/v1` (header `x-ident`, subprotocol `ucp4`, pinned
  device cert). While one is open, Find Anything sends `NL_PARSE {querySentence, model}` /
  `IMAGE_SEARCH {imgUri}` to the AI Key; without it Protect falls back to a console-local AI
  Controller gRPC (`127.0.0.1:7087`, only UNVR-G2/G2-PRO/Edge). Source: Protect modules
  `409403`, `323448`, `272635`, `359662`; the AI Key's `ui-slm-runtime-engine/slm_server.py`.
- **`NL_PARSE` reply**: `keyTags[{matchedWord, tags}]`, `objectTypes`, `startTime`/`endTime`/
  `timeTag`, `txtEmbed` (768-d CLIP; `model` `clip-ViT-L-14`), `exact_match`. Protect's **exact
  mode** searches by object label and returns 0 without one, so set `exact_match` only for a bare
  object word. Protect **caches replies 5 min per (model, text) and every hit resets the timer** —
  test changes with new wording.
- **Search runs on the AI Key's own PostgreSQL** (`ramDetections`, pgvector). Protect only writes
  a row for an object result that has `thumbnailTags` with `trackerID`, `keyMomentMs` (=
  `smartDetectObjects.detectedAt`) and a 768-d `imgEmbed`; it adds labels itself. Event-level tags
  and the description go to the **console's** DB instead. Similarity shown =
  `(cos − 0.08) / 0.32`; the UI slider filters after the SQL (default 50 %).
- **Tasks**: `recognizeKeyFrames` (video: `reqUrl` export — ask `format=mp4` instead of `ubv`;
  image/multipleImages for retroactive processing), `RequestAI` (`second_verifier_mlabel`;
  `:7445/generate-embeddings` and `:7968/describe` in deep mode), `enhanceImage`, `speechToText`.
  Inputs are console URLs fetched with the device cert + `x-ident`/`x-type`/`x-sysid` headers.
- **Boxes** in `thumbnailMeta`/`roiMeta`: `coord = [x, y, w, h]`, top-left, 0–1000 of the frame.
- **Feature gating**: `getInfo.featureFlags` (`supportPersonReId`, …) feed the console's
  in-memory `nvr.featureFlags.aiFeatureFlag`; the UI greys Person in Second Stage Verification
  without `supportPersonReId` (needs a browser reload after it changes).
- **Other replies Protect checks**: `diskInfo` → `{storageSize: "<num string>"}`,
  `networkStatus.linkSpeedMbps`, `getInfo.poeType` (`802.3bt`), `X-Sysid` header.
  `changeUserPassword` must be applied to the AI Key's PostgreSQL role.
- **Firmware**: Protect sends ucp4 `updateFirmware {uri, md5, timeoutMs}` and waits for a
  reconnect reporting the new version. Release builds are listed at
  `https://fw-update.ui.com/api/firmware?filter=eq~~product~~AI-Key&filter=eq~~platform~~OrinNX256`.
- **Retroactive processing** is one-shot per AI Key record (starts only from `not_started`).

## Traps

- open_clip: use `ViT-L-14-quickgelu` with `pretrained="openai"` (plain `ViT-L-14` silently uses
  the wrong activation).
- Night/IR crops sit ~0.2 cosine from matching text, under Protect's 50 %; stored embeddings blend
  0.1 × the object's tag text (`DIYKEY_CLIP_TAG_BLEND`).
- Protect's `service.js` is ~10 MB minified: use `grep -b -o -F` + `tail -c`; regex context like
  `.{0,150}` hangs. A de-minified per-module copy (`service-expanded/modules/`) is far easier.
- Camera attributes (e.g. `color: white`) can be wrong; CLIP/detector results disagreeing with
  them is not necessarily a bug.

## Lab (example setup — adjust for yours)

- Console "Console A" (UCG-Max, `192.0.2.1`, Protect 7.3.70); each Mac runs one instance on an
  interface alias (`192.0.2.18`) with its own MAC, device ID, GUID and PostgreSQL.
- Outside the repo: the detector ONNX (pi-port's `yolov9-s-320.onnx`), Protect's `ram_tags.csv`
  and firmware archives in `~/.diy-key/firmware/`.
- Protect extract (7.3.70) with de-minified modules: `<protect-extract>/`
  (`service-expanded/modules/`, `unifi-protect-ui-internal/`, `app/fixtures/ram_tags.csv`).
- Console logs worth tailing: `/srv/unifi-protect/logs/{aiprocessors,aiprocessors.db,service,errors,ds}.log`.
