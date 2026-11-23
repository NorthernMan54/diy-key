<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# diy-key: a UniFi Protect AI Key emulator on an Orange Pi

<div align="center">

<a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue" alt="License: AGPL-3.0-or-later"></a>

**A reverse-engineering / interoperability proof of concept that makes a stock
Orange Pi 5 Plus present itself to a real UniFi Protect console as genuine AI Key
hardware (device class `aiprocessors`, type `UP-AI-KEY`, sysid `0xa5f0`).**

**It speaks the vendor wire protocol only (UDP discovery, HTTPS adoption, the
mTLS `ucp4` WebSocket, on-demand AI task dispatch and result callbacks), with no
firmware modification, no controller database writes, and no Ubiquiti code or
binaries.**

</div>

> [!WARNING]
> Proof of concept, not production software. It runs discovery, an HTTPS
> adoption endpoint and the AI task plumbing, and expects a PostgreSQL instance
> for AI results. Use it only with hardware and a Protect console you own. It
> presents a synthetic device identity; do not run it on a network you do not
> control. See [Disclaimer](#disclaimer).

> [!NOTE]
> Not affiliated with, endorsed by, or connected to Ubiquiti Inc. "UniFi",
> "UniFi Protect", "AI Key" and "AI Port" are trademarks of Ubiquiti Inc. No
> Ubiquiti firmware or binaries are distributed here. Built with AI assistance
> under human review. See [Legal](#legal).

## Quick start

```bash
git clone https://github.com/hutchx86/diy-key && cd diy-key

sudo apt-get install -y python3-venv openssl
python3 -m venv ~/diykey-venv && . ~/diykey-venv/bin/activate
pip install -r diykey/requirements.txt
# Optional, only for the features you enable:
# pip install -r diykey/requirements-clip.txt   # DIYKEY_CLIP
# pip install -r diykey/requirements-lpr.txt    # DIYKEY_LPR
# pip install -r diykey/requirements-vlm.txt    # DIYKEY_VLM=transformers

cd diykey
sudo ~/diykey-venv/bin/python run_all.py --iface <your-interface>
```

Then click **Adopt** on the `AI Key` device in Protect. For a reboot-surviving
systemd instance with a fixed identity:

```bash
cd diykey
sudo python3 core/instance_manager.py create main --parent-iface <your-interface>
journalctl -u diy-key-instance-main -f
```

PostgreSQL setup (needed for the console to deliver AI results), migration of an
already-adopted identity, and the component-level detail live in
[`diykey/README.md`](diykey/README.md) and `implementation.md` in the private
workspace.

> [!NOTE]
> **macOS:** DIY-Key runs on a Mac against Protect 7.3 with real Basic Understanding tags, Second
> Stage Verification, retroactive processing, Find Anything search (the AI Key's nl-search
> connection + CLIP ViT-L/14) and a simulated firmware update. Build, configure and run it from
> [`docs/macos.md`](docs/macos.md); open work is in [`BACKLOG.md`](BACKLOG.md).

> [!NOTE]
> **Linux / Orange Pi:** the AI stages run on the deployed RK3588 runtime. Instance setup, the
> per-instance `diykey.env` for the detector/CLIP, and the current limits (CPU inference, shared
> PostgreSQL, no RKNN path yet) are in [`docs/linux.md`](docs/linux.md).

## At a glance

|  |  |
| --- | --- |
| **What** | Emulates the UniFi Protect AI Key device role |
| **Hardware** | Orange Pi 5 Plus (RK3588 NPU), verified live; any Linux host for the protocol stack |
| **Protocol / interface** | UBNT discovery UDP/10001; HTTPS :8080 adopt; mTLS WSS :7442 (`ucp4`); callbacks to console :7444; Postgres :5432; gRPC :50051 (stub) |
| **Language** | Python 3.11+ |
| **Status** | Proof of concept: adoption and the live RPC handshake verified against a real UNVR; face pipeline NPU-proven but not yet wired into dispatch |
| **License** | AGPL-3.0-or-later |

## How it works

```
Protect console                         DIY-Key (Orange Pi)
      │  UDP/10001 discovery (multicast)      │
      ├──────────────────────────────────────►│  stage1_discovery/beacon.py
      │  HTTPS :8080 GET /api/info            │
      │  HTTPS :8080 POST /api/adopt ────────►│  stage1_http/adopt_http.py
      │                                       │    └─ adopt_state.json (0600)
      │  mTLS WSS :7442 "ucp4" ◄─────────────►│  stage3_wss/ucp4_client.py
      │    getInfo / metrics / tasks          │    ├─ systemMetrics (RK3588 telemetry)
      │    recognizeKeyFrames / RequestAI     │    └─ result POSTs back to :7444
      │  PostgreSQL :5432 ◄───────────────────│  (console-owned schema + pgvector)
```

The device is an **AI Key**, not a camera: it never pulls RTSP video. The console
dispatches work on demand (a clip URL for `recognizeKeyFrames`, or a local
model-inference request for `RequestAI`), and the device reports results over the
console's internal API and/or Postgres. Identity is a persistent MAC/device-id
plus an EC P-256 client certificate whose fingerprint the console pins at
adoption; the token is single-use.

| Component | Role |
| --- | --- |
| `diykey/core/config.py` | One config/identity source (`diykey.cfg` + `DIYKEY_*`), iface/MAC/IP resolution, atomic state |
| `diykey/core/instance_manager.py` | One systemd unit per emulated device (macvlan or physical), reboot persistence |
| `diykey/stage1_discovery/beacon.py` | UBNT discovery responder (real TLV set/order) |
| `diykey/stage1_http/adopt_http.py` | HTTPS control API, adoption persistence, cert generation |
| `diykey/stage3_wss/ucp4_client.py` | Outbound mTLS `ucp4` client, RPC dispatch, task callbacks |
| `diykey/stage3_wss/system_stats.py` | RK3588 CPU/GPU/NPU/RAM metrics for `systemMetrics` |
| `diykey/stage4_modelserver/model_server.py` | `ui-infer-gateway`-shaped inference endpoint on :7788 |
| `diykey/stage2_grpc/` | Stub `ProtectAPI` gRPC server on :50051 (not yet confirmed dialed) |
| `diykey/stage5_faceid/` | RKNN face detection + embedding pipeline (NPU-proven, not wired) |

## Features

- Live-verified adoption flow: discovery → flat `/api/adopt` → mTLS `ucp4`.
- Exact AI-Key RPC set: `getInfo`, `networkStatus`, `changeUserPassword`,
  `systemMetrics`, `getTaskQueueInfo`, `recognizeKeyFrames`, `RequestAI`; unknown
  actions are acked and logged for capture.
- Tokenless reconnect with a pinned cert, single-use adoption tokens, and
  re-adoption watcher (readopt/unadopt forces a clean session).
- Real RK3588 telemetry (CPU, GPU, memory, plus NPU for diagnostics), sampled every
  second and averaged over 30 s/300 s like the firmware, returned as 0-1 fractions
  in `systemMetrics`.
- Task plumbing for RAM/reverification results, multipart `ram` upload and the
  console's `/internal/aiprocessors/*` callbacks.
- Shared config/state layer, per-instance systemd units, and 482 unit tests.
- Wired AI work in `stage3_wss/`: SFace face recognition, fast-alpr plates, GFPGAN
  enhancement, VLM summaries, OSNet re-ID and per-camera scene memory.
- Face detection (`RetinaFace_mobile320`) and embedding (`w600k_mbf`) running on
  the RK3588 NPU; see the limitation below before using the embeddings.

## Requirements

- Orange Pi 5 Plus (RK3588) for the verified path; the protocol stack runs on any
  Linux host. Root is needed for interface scoping/macvlan (the listeners are on
  unprivileged ports).
- Python 3.11+, `python3-venv`, `openssl`.
- Optional: Rockchip `rknn-toolkit-lite2` + `librknnrt` 2.3.2 for the face
  pipeline (`diykey/requirements.txt` explains the wheel).
- PostgreSQL 15 + `pgvector` for the console's AI-results channel.

## Repository layout

```
.
├── diykey/                 # source: stages, core, tests, diykey.cfg
│   ├── core/               # config/state + systemd instance manager
│   ├── stage1_discovery/   # UDP/10001
│   ├── stage1_http/        # HTTPS/8080
│   ├── stage2_grpc/        # ProtectAPI stub + generated bindings
│   ├── stage3_wss/         # ucp4 client + RK3588 telemetry
│   ├── stage4_modelserver/ # inference endpoint stub
│   ├── stage5_faceid/      # RKNN face pipeline
│   └── tests/              # unittest suite
├── scripts/                # deploy/SSH helpers
├── rknn_convert/           # host-side model conversion
├── CREDITS.md
└── LICENSE
```

## Install and usage

The full walkthrough (venv, generated bindings, PostgreSQL/pgvector, the macvlan
instance, and migrating an already-adopted identity without changing the pinned
cert) is in [`diykey/README.md`](diykey/README.md). The short version:

```bash
cd diykey
sudo python3 run_all.py --iface <iface> --hostname diy-key   # foreground
sudo python3 core/instance_manager.py list                  # instances + live IPs
```

The adopt token, TLS keypair and per-instance state live under
`diykey/stage1_http/` for a single instance, or `diykey/instances/<name>/` with the
manager. The key is mode 0600; never commit the state directory.

## Configuration

`diykey/diykey.cfg` holds `[identity]` (MAC, hostname, platform, sysid, firmware,
device-id, guid), `[network]`, `[console]` and `[logging]`. Every value has a
`DIYKEY_*` environment override and CLI flags; CLI wins, then env, then the file.
Do not change `device_id`/`guid`/MAC once a console has adopted the device, and
do not regenerate `server.crt`/`server.key` without un-adopting first; the
console pins the fingerprint.

## Verification

No hardware needed:

```bash
cd diykey && python3 -m unittest discover -s tests -v   # 482 tests
```

Verified live on an Orange Pi 5 Plus against a real UNVR (2026-09-07): discovery,
adoption, the `ucp4` handshake and RPCs, and `recognizeKeyFrames`/`RequestAI`
placeholder results accepted (HTTP 200). The protocol facts are documented in the
private `protocol.md` (§1 discovery, §2 adoption, §3 ucp4, §5 inference contracts,
§6 Postgres, §7 face recognition, §7.1 face-DB embedding space).

Verified on the board since 2026-10-30: systemd instance, adoption, RAM tagging
and event delivery. Not yet verified: CLIP/Find Anything on the board, the gRPC
service being dialled by a console, and face results delivered end to end.

## Known limitations

- **Face embeddings are the wrong space for Protect.** Protect's face library is
  128-d SFace; `stage5_faceid/` uses 512-d MobileFaceNet. Detection/alignment are
  fine; the embeddings cannot be enrolled as-is (`protocol.md` §7.1).
- RAM tagging and second-stage verification need `DIYKEY_RAM_MODEL`; without it
  the result is a placeholder. LPR, face recognition, face enhancement and VLM
  descriptions are implemented but need their models and `DIYKEY_*` switch.
  Speech-to-text is not implemented.
- The gRPC `ProtectAPI` server is a stub; no console has been observed dialling it.
- Video-frame delivery is unsolved for the AI-Key path (on-demand clip fetch is
  the likely route; not implemented).
- Open work is tracked in [`BACKLOG.md`](BACKLOG.md).

## Credits

Shared infrastructure is derived from the sibling **pi-port** project (same
author, AGPL-3.0-or-later): config/state, instance manager, discovery, adopt HTTP,
ucp4 framing/reconnect, run_all supervision and the test harness. The exact file
list and third-party attributions are in [`CREDITS.md`](CREDITS.md). Face models:
`RetinaFace_mobile320` from `airockchip/rknn_model_zoo`; `w600k_mbf` from
insightface's `buffalo_sc` pack, converted locally with `rknn-toolkit2`.

<a id="legal"></a>
<details>
<summary><strong>Legal</strong></summary>

This project is an independent interoperability experiment. It is not
affiliated with, endorsed by, sponsored by, or connected to Ubiquiti Inc.
"UniFi", "UniFi Protect", "AI Key" and "AI Port" are trademarks of Ubiquiti Inc.,
used here only to describe the hardware this project interoperates with.

No Ubiquiti firmware, binaries, model files or other proprietary code are
distributed in this repository. The `.proto` files under `diykey/proto/` are
interface definitions used for interoperability; no vendor implementation is
included. All trademarks remain the property of their respective owners.

This project was built with AI assistance under human review. Treat every claim
as something to verify on your own hardware.

</details>

<a id="disclaimer"></a>
<details>
<summary><strong>Disclaimer</strong></summary>

This is a proof of concept. It is not production software and comes with no
warranty of any kind, express or implied. It manipulates network interfaces,
DHCP leases, self-signed certificates and device identities; misconfiguration
can disrupt your network, your Protect console or the device it runs on. The
authors are not liable for any damage or loss arising from its use.

Use it only on hardware and consoles you own and control. Do not use it to
impersonate devices on networks you do not own. You are responsible for
complying with the laws and terms that apply to you.

</details>

## Security

- The adopt token and TLS private key are written mode 0600 and are gitignored;
  keep the state directory out of version control and backups you share.
- The console does not validate the device certificate, and the device does not
  validate the console certificate (matching the real firmware); transport
  security here is identity pinning, not a trusted PKI.
- The emulator accepts the console's adoption POST without authentication, like
  the real device. Do not expose it to an untrusted network.
- The default PostgreSQL credentials are the console's own defaults and are only
  suitable for a lab.

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE) and [`CREDITS.md`](CREDITS.md).
