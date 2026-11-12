# DIY-Key on macOS (Protect 7.3)

How to build, configure and run DIY-Key as an AI Key on a Mac, as developed on the `MacOS`
branch against UniFi Protect 7.3.70 on a UCG-Max. One Mac can run several instances; each is an
AI Key with its own IP address, identity, state and PostgreSQL.

## What it does

| Protect feature | DIY-Key | Notes |
|---|---|---|
| Adoption, settings sync, firmware state | ✅ | `getInfo` advertises the AI Key feature flags (person re-ID, face, LPR, RAM, …) |
| Basic Understanding (`recognizeKeyFrames`) | ✅ | Clip fetched as MP4, frames tagged by a COCO ONNX detector, per-object CLIP embeddings |
| Retroactive processing | ✅ | Image tasks (`multipleImages`); one-shot per adoption |
| Second Stage Verification (`RequestAI`) | ✅ (offline-tested) | Box (0–1000 `[x,y,w,h]`) matched against the detector |
| Find Anything (typed search) | ✅ | The AI Key's nl-search WebSocket; CLIP ViT-L/14 text/image embeddings |
| Firmware update | Simulated | Real file stored, version pinned, nothing installed |
| Face / LPR / face enhancement / speech-to-text / VLM descriptions | ❌ | See `BACKLOG.md` |

## Prerequisites

```bash
brew install postgresql@17 pgvector ffmpeg python@3.14
```

- A COCO object detector exported to ONNX in the Ultralytics anchor-free layout
  (`[1, 84, N]` output), e.g. Frigate's `yolov9-s-320.onnx`.
- Optional, for Find Anything: ~2 GB of disk for torch and the CLIP ViT-L/14 weights (downloaded
  from Hugging Face on first use).
- Optional, for query tags: Protect's `app/fixtures/ram_tags.csv` from your console's Protect
  install (`/usr/share/unifi-protect/app/fixtures/ram_tags.csv`).

## Build

From the repository root:

```bash
python3 -m venv venv
venv/bin/pip install -r diykey/requirements.txt     # includes onnxruntime, torch, open_clip
venv/bin/python -m unittest discover -s diykey/tests   # 3 test_run_all errors are macOS-only
```

## Configure an instance

Everything for instance `<name>` lives in `~/.diy-key/<name>/` (see `scripts/diykey_paths.sh`):

```
diykey.env        settings (DIYKEY_* only)
state/            adopt state, server.crt/key (pinned by Protect), fw_version.json, clips/
pg/               the instance's PostgreSQL cluster
logs/             diykey.log (rotated), postgres.log
restart           touch to restart DIY-Key
```

1. **Pick a free IP** on the console's network and add it to the interface (not persistent):
   ```bash
   sudo ifconfig en0 alias 192.0.2.18 255.255.255.255
   ```
2. **Write `~/.diy-key/<name>/diykey.env`** from `scripts/diykey.env.example`: a new MAC
   (`FCECDA` + 6 random hex digits), device ID and GUID (`uuidgen | tr A-Z a-z`), the bind IP and
   the console address. Keep the identity fixed once adopted.
3. **Create the instance's PostgreSQL** (listens on the bind IP, admits only the console over SSL):
   ```bash
   DIYKEY_INSTANCE=<name> scripts/diykey_pg_init.sh
   ```

### Settings (`diykey.env`)

| Key | Purpose |
|---|---|
| `DIYKEY_IFACE`, `DIYKEY_MAC`, `DIYKEY_DEVICE_ID`, `DIYKEY_GUID`, `DIYKEY_HOSTNAME` | Identity (required except hostname) |
| `DIYKEY_BIND_IP` | The instance's own address: listeners, outbound ucp4 and nl-search, PostgreSQL |
| `DIYKEY_CONSOLE_HOST` | Console address (tokenless reconnect, PostgreSQL access rule) |
| `DIYKEY_RAM_MODEL` | COCO ONNX detector for tags, verification and crops; also advertises Recognize Anything and retroactive processing. Unset = no RAM result |
| `DIYKEY_CLIP=1` | CLIP ViT-L/14 embeddings: per-object rows for search, query embeddings |
| `DIYKEY_FACE`, `DIYKEY_LPR`, `DIYKEY_FACE_ENHANCE` | `1` to advertise face recognition, licence plates, face enhancement (default off) |
| `DIYKEY_VLM` | AI summary backend, `mlx` or `transformers`; unset, `off` or `0` = not advertised |
| `DIYKEY_VLM_MODEL` | AI summary model: an mlx-community repo (`mlx`) or a local InternVL2-style checkpoint path (`transformers`) |
| `DIYKEY_DEEP_MODE` | `1` (with `DIYKEY_VLM`) advertises `supportDeepMode`/`supportVlm`. Protect then sends `modelMode: deep` when Deep Understanding is on for the NVR, turns RAM off on this key, and dispatches `RequestAI :7968/describe`; the key replies with a VLM description of a representative frame, `labels` from the RAM tagger if `DIYKEY_RAM_MODEL` is set, and (with `DIYKEY_CLIP=1` and `DIYKEY_DESC_EMBED=1`) a CLIP text embedding. `getInfo` reports `aiMode: deep` only when this is on. Person re-ID needs a 1024-d re-ID model (CLIP is 768-d), so `:7445/generate-embeddings` tasks are answered as failed |
| `DIYKEY_FACE_MODELS` | Folder holding the YuNet and SFace `.onnx` files (default `<diykey>/models`) |
| `DIYKEY_FACE_ENHANCE_MODEL` | GFPGAN v1.4 ONNX for face enhancement; unset = the original face is uploaded |
| `DIYKEY_LPR_THRESH`, `DIYKEY_LPR_MIN_PX` | Plate-detector score floor (default 0.5) and minimum plate width in pixels (60) |
| `DIYKEY_FACE_MIN_PX` | Narrowest face sent to Protect, in frame pixels (default 40, 0 keeps all) |
| `DIYKEY_REID_MODEL` | OSNet-AIN x1.0 MSMT17 checkpoint: person re-ID vectors and tap-to-find |
| `DIYKEY_TASK_WORKERS`, `DIYKEY_TASK_QUEUE_MAX`, `DIYKEY_TASK_DROP_AGE_S` | Task pool: workers (default 1), max queued (200; full = drop), max wait before discard (900 s) |
| `DIYKEY_PREFETCH_WORKERS` | Threads that download each event's clip and grab its frames as soon as the task arrives (default 2, `0` = off, download on the task worker). The task worker then only runs the models, so a slow download doesn't hold up the next event. In the timing line, `clip=`/`frames=` are then the prefetch times and `fetchwait=` is what the worker still waited |
| `DIYKEY_VLM_QUEUE` | `0` to write automatic summaries on the main task worker. Default on: they have their own queue and worker, so tags aren't held up |
| `DIYKEY_TRUST_CLAIMED_TYPE` | `1` to tag every object crop with the type Protect claims (person/vehicle/animal). Default off: with RAM++ loaded, a crop where neither COCO nor RAM++ sees that kind of object (and, for a vehicle, the camera gave no vehicle type) doesn't get the type as a tag, so alarm rules and tag search don't match a crack called "animal". Protect's own label for the object is unchanged |
| `DIYKEY_CLIP_VETO` | When Protect says "animal" but CLIP is at least this sure the crop is something else (a crack, stain, shadow, stump, bin, person, car...), the object loses its animal and species tags (default 0.9, `0` = off; needs `DIYKEY_CLIP=1`). Protect's own label is unchanged |
| `DIYKEY_SPECIES_MIN_PX` | Animal crops keep RAM++'s species (cat, dog, deer...) so Find Anything can match them, but only when the event's whole frames name the same species. Without whole frames (retroactive image tasks), the crop's short side must reach this many pixels instead (default 160) |
| `DIYKEY_WORD_LINKER` | Optional path to the firmware's `ui_slm_word_linker.json` (ui-slm-runtime-engine, outside the repo): maps query words to RAM tags (`pickup` → truck) at weight ≥ 0.8. Regular plurals (`boxes` → box) work without it |
| `DIYKEY_RAM_TAGS` | Path to Protect's `ram_tags.csv` (query tags; otherwise a built-in list of names checked against it) |
| `DIYKEY_NL_SEARCH` | `0` to not open the nl-search connection (default on) |
| `DIYKEY_CLIP_TAG_BLEND` | Weight of the tag text blended into stored embeddings (default 0.1, 0 = off). The same vector feeds search and AI Key Advanced alarm scores: higher = more alarms fire on crops matching their tags, but wrong tags inflate scores too |
| `DIYKEY_VLM_EVENT_CONTEXT` | `0` to summarise the whole frame as the firmware does. Default on: automatic summaries describe a crop 3x the triggering object's box (1.5x for a vehicle at the frame edge) and say what triggered the event |
| `DIYKEY_SCENE_MEMORY` | `1` to learn per camera what is always in the picture, so a parked car stops appearing in tags and summaries and changes get reported (default off; see below) |
| `DIYKEY_SCENE_VEHICLE_*`, `DIYKEY_SCENE_ANIMAL_*`, `DIYKEY_SCENE_PERSON_*` | Scene-memory thresholds, starting guesses: `_EVENTS` (3), `_MINUTES` (vehicle 30, others 10), `_IOU` (0.6, 0.2), `_SIM` (0.85, animal 0.75, person 0.7); animal and person also `_SCORE` (0.7, 0.6) and `_EXPIRE_MINUTES` (60, 15) |
| `DIYKEY_SCENE_FIXTURE_DAYS`, `_FIXTURE_EVENTS`, `_FIXTURE_IOU`, `_MISSING_EVENTS`, `_TRIGGER_OVERLAP`, `_APPEARANCE_WEIGHT`, `_KEEP_DAYS` | Same-look-and-spot age for a fixture (7 days, at least 7 events, IoU 0.6); events missing before a move is reported (2); overlap with the trigger box that isn't evidence (0.3); appearance weight for animals and people (0.75); how long unconfirmed objects are kept (3 days) |
| `DIYKEY_FW_VERSION`, `DIYKEY_DISCOVERY_FW_VERSION` | Factory firmware; a simulated update pins newer in `state/fw_version.json` |
| `DIYKEY_FW_DIR`, `DIYKEY_FW_DOWNLOAD`, `DIYKEY_FAKE_UPDATE_S` | Where offered firmware is kept (default `~/.diy-key/firmware`), whether, and how long the fake install takes |
| `DIYKEY_PG`, `DIYKEY_PG_BIN` | `0` to run without the instance's PostgreSQL; Homebrew `postgresql@17` bin dir |
| `DIYKEY_DEBUG`, `DIYKEY_LOG_MAX_MB`, `DIYKEY_LOG_KEEP` | Logging |

AI Key Advanced alarms (Alarm Manager rules with a natural-language sentence and a threshold) need
`DIYKEY_CLIP=1`, `DIYKEY_RAM_MODEL`, `DIYKEY_RAM_TAGS` and the nl-search connection; startup logs
one warning naming whatever is missing. Protect parses a rule's sentence once (at first use) and
caches the result (tags and text embedding) on the rule; a rule saved while these were missing may
keep an empty tag list and never fire, so recreate it after fixing them.

Scene memory (`DIYKEY_SCENE_MEMORY=1`) keeps `state/scene_memory/<camera>.json`. It needs
`DIYKEY_CLIP=1` for vehicles and animals and `DIYKEY_REID_MODEL` for people. It learns only from
frames at events, so a change is noticed at the next event, and a moved camera resets what it has
learned, reporting the old objects as gone once. People are never removed from tags, however long
they stay. A fixture (7 days unchanged) is ignored and dropped silently, so a car parked that long
no longer gets a "has moved" sentence. Delete a camera's file to start it over.

## Run

```bash
DIYKEY_INSTANCE=<name> scripts/diykey_service.sh      # in a terminal; Ctrl-C stops it and its PostgreSQL
touch ~/.diy-key/<name>/restart                        # restart with new code/settings
tail -f ~/.diy-key/<name>/logs/diykey.log
```

The service refuses to start until the bind IP is on the interface, starts the instance's
PostgreSQL, restarts DIY-Key if it exits, and exports the PostgreSQL socket so Protect's rotated
database password is applied. Then click **Adopt** on the AI Key in Protect.

That run lasts as long as its terminal. To keep an instance running, across reboots too, install
it as a root launchd daemon, which also adds the bind IP and starts PostgreSQL (as the owner of
`pg/`, since PostgreSQL refuses root). Stop any terminal run first:

```bash
sudo DIYKEY_INSTANCE=<name> scripts/macos_launchd.sh install
DIYKEY_INSTANCE=<name> scripts/macos_launchd.sh status
```

`touch ~/.diy-key/<name>/restart` still works without sudo; the script's header lists the rest.

## Check it works

- `logs/diykey.log`: `WSS CONNECTED`, `nl-search connected to wss://<console>:7443/wss/nl-search/v1`,
  and per event `clip for event …`, `event …: <description>`, `POST …/recognize-anything -> 200`.
- Search rows: `psql -h ~/.diy-key/<name>/pg -d unifi-protect -c 'select count(*) from "ramDetections"'`.
- Find Anything: `NL_PARSE '<query>' -> tags … types … embed 768` in the log. Protect caches each
  query's reply for 5 minutes (every repeat resets it), so use new wording after a change.
- On the console: `/srv/unifi-protect/logs/aiprocessors.log`, `aiprocessors.db.log`, `ds.log`.

## Move to another Mac

**New pairing** (a fresh AI Key on the console; e.g. moving to a replacement Mac):

1. In Protect, **Unadopt/Forget** the old AI Key while it is still online, then stop the old
   service (Ctrl-C in its terminal stops DIY-Key and its PostgreSQL). Nothing in
   `~/.diy-key/<name>/` needs to come along: a new pairing gets a new identity, cert and database.
2. On the new Mac: clone, check out `main`, install the prerequisites and build (above).
3. Copy the files that are not in the repo (vendor material or weights, never committed):
   - the detector ONNX (e.g. `yolov9-s-320.onnx`) → path in `DIYKEY_RAM_MODEL`;
   - Protect's `ram_tags.csv` → path in `DIYKEY_RAM_TAGS`;
   - optional: `~/.diy-key/firmware/` (only to skip re-downloading offered firmware);
   - the CLIP weights download again on first use (or copy `~/.cache/huggingface/hub/`).
4. Configure an instance (above) with a **new** MAC, device ID, GUID and a free IP; don't reuse
   the old identity, or the console may treat it as the old, pinned device.
5. Run, then adopt in Protect. Retroactive processing starts again for the new AI Key record.

**Same identity** (keep the adoption): stop the old instance, copy all of `~/.diy-key/<name>/`
(`state/` holds the cert Protect pinned, `pg/` the database Protect migrated), keep
`DIYKEY_BIND_IP` (Protect remembers the address) or update it in Protect, and fix the paths in
`diykey.env`.

## Notes

- Firmware Protect offers is kept in `~/.diy-key/firmware` with its md5; never commit vendor
  firmware. `<aikey-fw>/fetch.sh` (outside the repo) fetches any version from
  `fw-update.ui.com` the way Protect does.
- PostgreSQL statement logging may be on for debugging:
  `psql -h ~/.diy-key/<name>/pg -d postgres -c "ALTER SYSTEM RESET log_statement" -c "SELECT pg_reload_conf()"`.
