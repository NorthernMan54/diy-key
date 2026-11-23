# DIY-Key — UniFi Protect "AI Key" emulation on an Orange Pi 5 Plus / RK3588

**STATUS: adoption works end-to-end, live, against a real UniFi Protect
console.** The Pi shows up in Protect as `AI Key`, adopts cleanly, and holds a
stable connection. The full confirmed wire-level writeup and project history
live in the project's private research notes (kept out of this repo); this
README describes the resulting architecture.

## Real protocol (confirmed live, not guessed)

Turned out to be architecturally close to the already-reverse-engineered
UniFi Protect **Chime** protocol — not the SSH `mca-ctrl` mechanism found
in AI Key's own firmware, which appears to be for a different/unused
adoption path.

1. **UDP/10001 discovery** — the classic UBNT L2 discovery protocol,
   generic across device classes. The `platform` TLV field (`0x0C`) must
   be a SKU string the controller's catalog recognizes; `UP-AI-KEY` works.
2. **HTTPS control API on the device, port 8080** — `GET /api/info`
   (polled ~60s), `POST /api/adopt` (on clicking Adopt; body includes
   `hosts`, `token`, `protocol:"wss"`, `consoleId`, etc. — same shape as
   the chime spec). Other `/api/<action>` posts are acknowledged with 200.
3. **Outbound mTLS WebSocket to `hosts[0]` (`<console-ip>:7442`)** — same
   binary `ucp4` envelope as chime (8-byte record header + JSON payload,
   one head record + one body record per message). RPCs specific to
   `aiprocessors` (not in the chime spec) discovered live: `getInfo`,
   `networkStatus`, `changeUserPassword`, `sshService`, `firmwareState`,
   `diskInfo`, `updateTimezone`, `changeAiInferAgentSettings` (the real
   feature toggles: face recognition, LPR, RAM, STT, region),
   `setConsoleInfo`, `updateLcmSettings`, `getTaskQueueInfo` (a ~5s
   keepalive) and `systemMetrics` (backs the System Performance tab).
   Exact response *schemas* for `systemMetrics` and `getTaskQueueInfo`
   were confirmed by reading live Zod validation errors out of the real
   controller's own `errors.log` and cross-referencing its bundled
   `service.js` source — see the private protocol notes §3 for the exact shapes.
4. **Direct PostgreSQL connection from the console to the device, port
   5432** — confirmed to be the actual AI-results delivery channel (not
   any RPC). See "PostgreSQL setup" below.

The local gRPC service (`stage2_grpc/`, byte-compatible with the real
`.proto` contract) does **not** appear to be dialed by the console during
this flow — it may serve a different purpose (e.g. detections streaming
once real inference exists) not yet exercised.

## Components

- **`core/config.py`** — one source of truth for identity/config (`diykey.cfg` + `DIYKEY_*`
  env overrides): platform/sysid/firmware/device-id/guid, iface/MAC/IP resolution,
  `--bind-ip`/`--mac`, log redaction, and the atomic state writer.
- **`core/instance_manager.py`** — one systemd unit per emulated AI Key (macvlan +
  dhclient or static IP), each with a fixed MAC/device-id and its own state dir. Survives
  reboot; `create|destroy|refresh|list`.
- **`stage1_discovery/beacon.py`** — UDP/10001 discovery responder. Sends the real field
  set/order (`protocol.md` §1), reports `is_managed` from the live adopt state, and joins
  the controller's multicast group.
- **`stage1_http/adopt_http.py`** — HTTPS `:8080` server (`/api/info`, `/api/adopt`);
  persists the adopt payload atomically (0600) to `<state-dir>/adopt_state.json` and
  clears it on `/api/readopt`.
- **`stage3_wss/ucp4_client.py`** — outbound `ucp4` WebSocket client that completes
  adoption and handles the ongoing RPC traffic. Falls back to a tokenless reconnect to a
  known console host if the device restarts after already being adopted server-side, and
  watches `adopt_state.json` for any token change to force a clean reconnect. Framing is
  the real 8-byte header with a **u32** length (`protocol.md` §3).
- **`stage3_wss/system_stats.py`** — real RK3588 hardware stats (CPU via
  `/proc/stat`, GPU via `/sys/devices/platform/fb000000.gpu/utilisation` or
  `/sys/class/devfreq/*.gpu/load`, NPU via `/sys/kernel/debug/rknpu/load`, RAM,
  storage). A 1 s sampler publishes 30 s block averages for the first five
  minutes and 300 s afterwards, matching `aikey_monitor.py`; `systemMetrics`
  serves that cached average as 0-1 fractions.
- **`stage2_grpc/`** — stub `ProtectAPI` gRPC server on `:50051`, real
  `.proto` contract. Kept running; not yet confirmed to be used by the
  live flow.
- **`proto/`** — verbatim `.proto` files from the real `ui-uaibd` package.
- **`tests/`** — 482 stdlib `unittest` cases covering the shared config/state layer,
  discovery TLVs, ucp4 framing/handlers, adopt persistence/certs, instance units and
  `run_all` wiring. `cd diykey && python3 -m unittest discover -s tests -v`.
- **`stage5_faceid/`** — real face detection + embedding on the RK3588 NPU, standalone-tested
  (not yet wired into `run_all.py` or the ucp4/model-server RPC handlers — tracked in `BACKLOG.md`):
  - `face_detect.py` — `FaceDetector`, wraps a RetinaFace_mobile320 RKNN model (converted from
    `airockchip/rknn_model_zoo`'s example) via `rknnlite`. ~9ms/frame.
  - `face_embed.py` — `FaceEmbedder` + `align_face()`, wraps a MobileFaceNet RKNN model
    (`w600k_mbf`, insightface's `buffalo_sc` pack, converted fp16/unquantized) plus the standard
    insightface 5-point alignment. Confirmed on a 6-face group photo: same-face similarity 1.000,
    cross-face pairs near 0. **Caveat:** this is a 512-d embedding; Protect's face library is
    128-d SFace space (see the private protocol notes §7.1), so it is not enrolment-compatible as-is.
  - `pipeline.py` — `FacePipeline`, combines the two into one `process(image) -> [{bbox, score,
    embedding}]` call. This is the entry point to use once real frames/dispatch are wired in.
  - `models/` — test images; the `.rknn` models are converted locally and gitignored.

## Quick start on the Orange Pi

Commands below run from `diykey/` (the repo's source directory) unless noted.

```bash
sudo apt-get install -y python3-pip python3-venv openssl
python3 -m venv ~/diykey-venv
source ~/diykey-venv/bin/activate
pip install -r requirements.txt
# Optional: requirements-clip.txt (DIYKEY_CLIP), requirements-lpr.txt (DIYKEY_LPR),
# requirements-vlm.txt (DIYKEY_VLM=transformers).

# The committed generated bindings need grpcio>=1.83.1 / protobuf>=7.35.1
# (requirements-dev.txt pins them). Regenerate only if the .proto files change:
bash stage2_grpc/generate_pb2.sh

# Review diykey.cfg (identity, console host) and run everything. Needs root for
# interface scoping (SO_BINDTODEVICE) / macvlan; the ports themselves are unprivileged.
sudo ~/diykey-venv/bin/python run_all.py --iface eth0 --hostname diy-key
```

Or let systemd run it as a fixed-identity instance that survives reboot:

```bash
sudo python3 core/instance_manager.py create main --parent-iface <parent-interface>
python3 core/instance_manager.py list
# logs: journalctl -u diy-key-instance-main -f
```

To migrate an **already-adopted** device, keep its cert/token and pass the old identity —
otherwise the console sees a new fingerprint. Exact steps are in the private `implementation.md`
("Migrating the existing adopted board").

Then trigger "Adopt" from the Protect console UI. Watch the journal (or the terminal) —
you'll see the discovery probes, the `/api/adopt` POST with its real `hosts`/`token`, the
WSS connect, and the RPC handshake.

Automated checks (no hardware needed; from the repo root):

```bash
python3 -m unittest discover -s diykey/tests -v
```

## PostgreSQL setup (required for the console to fully "operate" the device)

Confirmed exact connection parameters (from the real controller's
`service.js`): host = this device's IP, port `5432`, user `unifi-protect`,
password `unifi-protect` (rotated later via the `changeUserPassword` RPC),
database `unifi-protect`, SSL required but not certificate-validated.
**The console migrates its own schema into this database** — you only
need to stand up an empty, correctly-configured instance:

```bash
sudo apt-get install -y postgresql postgresql-server-dev-<major>
# <major> is your release's PostgreSQL major (Debian 13: 17, Ubuntu 24.04: 16)
# SSL is on by default via Debian's snakeoil cert -- sufficient, since the
# controller doesn't validate it. Just open it up to the LAN:
sudo sed -i "s/^#listen_addresses = 'localhost'/listen_addresses = '*'/" /etc/postgresql/<major>/main/postgresql.conf
echo "hostssl unifi-protect unifi-protect <your-lan-cidr> scram-sha-256" | sudo tee -a /etc/postgresql/<major>/main/pg_hba.conf
sudo -u postgres psql -c "CREATE ROLE \"unifi-protect\" WITH LOGIN PASSWORD 'unifi-protect' SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE \"unifi-protect\" OWNER \"unifi-protect\";"

# pgvector isn't packaged for Debian/aarch64 -- build from source:
git clone --depth 1 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
sudo -u postgres psql -d unifi-protect -c "CREATE EXTENSION IF NOT EXISTS vector;"
# (SUPERUSER above is needed because vector.control lacks `trusted = true`,
# so even IF NOT EXISTS needs superuser to satisfy the console's own
# unconditional CREATE EXTENSION during migration.)

sudo systemctl restart postgresql
```

Then trigger a fresh WSS reconnect (kill and restart `ucp4_client.py`, or
just wait — the console retries the PG connection with backoff on every
reconnect) and the console's migrator creates the real tables:
`ramDetections` (`embedding vector(768)`, `reidEmbedding vector(1024)`),
`ramDetectionLabels`, `smartDetectSessionObjects`
(`reidEmbedding vector(512)`), `smartDetectSessionsSearch`
(`descEmbedding vector(384)`, free-text `description`). Full column list
in the private protocol notes §6 — this is the real schema for delivering AI
results into Protect.

## Known open questions / next steps

`BACKLOG.md` tracks the current list (facial recognition is the active thread;
NPU inference, video-frame delivery and the gRPC dial question are open).
Adopt-state persistence and reboot survival are handled by `core/config.py` +
`core/instance_manager.py`.

## Safety / scope note

This only talks to hardware and a console **you own**, using vendor-default
credentials **you are entitled to use on your own gear**, purely to
interoperate with your own UniFi Protect install. Nothing here targets
third-party systems.
