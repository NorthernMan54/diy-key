<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# DIY-Key on Linux (Orange Pi 5 Plus / RK3588)

How to run the AI work on the deployed runtime: Debian/Ubuntu on an Orange Pi 5 Plus, one
systemd unit per instance, a system PostgreSQL with pgvector. The macOS workflow is unchanged;
see [`macos.md`](macos.md).

## What runs where

| | macOS (`diykey_service.sh`) | Linux (this doc) |
|---|---|---|
| Process supervision | shell wrapper, `~/.diy-key/<name>/` | systemd, `core/instance_manager.py` |
| Address | `ifconfig <if> alias` | macvlan child interface (or a real NIC) |
| State | `~/.diy-key/<name>/state/` | `diykey/instances/<name>/` |
| PostgreSQL | per-instance cluster in `pg/` | system cluster on `:5432` (role/db `unifi-protect`) |
| Rotated PG password | `DIYKEY_PG_SOCKET=<name>/pg` | automatic: psql as `postgres` over `/var/run/postgresql` |
| Settings file | `~/.diy-key/<name>/diykey.env` | `<state-dir>/diykey.env` (unit `EnvironmentFile`) |

The AI code itself is platform-neutral: the COCO detector runs through `onnxruntime`
(CPUExecutionProvider), and CLIP uses whichever of MPS/CUDA/CPU torch finds. Nothing in
`ram_tagger`, `clip_embed`, `nl_search`, `second_verifier`, `clip_fetch` or `firmware` is
macOS-specific.

## Dependencies

```bash
sudo apt-get install -y python3-venv ffmpeg postgresql isc-dhcp-client
# Also install postgresql-server-dev-<major> and postgresql-<major>-pgvector for your
# release (Debian 13 is PG17, Ubuntu 24.04 is PG16). The DHCP client is only needed
# when create runs without --ip.

python3 -m venv ~/diykey-venv
. ~/diykey-venv/bin/activate
pip install -r diykey/requirements.txt
# Optional AI stacks, only for the features you enable:
# pip install -r diykey/requirements-clip.txt   # DIYKEY_CLIP
# pip install -r diykey/requirements-lpr.txt    # DIYKEY_LPR
# pip install -r diykey/requirements-vlm.txt    # DIYKEY_VLM=transformers
```

`onnxruntime` (RAM tags, verification) is a normal aarch64 wheel. `torch` + `open_clip_torch`
live in `requirements-clip.txt` because they are large and CPU CLIP is slow; leave Find
Anything off unless you want it on the board.

## PostgreSQL for AI results

Protect connects directly to the key's database and writes `ramDetections` and the session
search tables there. A RAM-only instance can skip this; Find Anything, alarms and tap-to-find
need it. Use the same role and database as the real key, then allow the console:

```bash
sudo -u postgres createuser --superuser unifi-protect
sudo -u postgres createdb -O unifi-protect unifi-protect
sudo -u postgres psql -c "ALTER ROLE \"unifi-protect\" PASSWORD 'unifi-protect';"
# /etc/postgresql/<major>/main/postgresql.conf: listen_addresses = '*'
# /etc/postgresql/<major>/main/pg_hba.conf:
#   hostssl unifi-protect unifi-protect <console-ip>/32 scram-sha-256
sudo systemctl restart postgresql
```

The console runs its own schema migration. The default password is the real device's default;
change it on both sides if the network is not isolated.

## Configure an instance

Create the unit as before (identity is fixed at creation):

```bash
cd diykey
sudo "$HOME/diykey-venv/bin/python" core/instance_manager.py create main --parent-iface <board-nic> \
  --mac FC:EC:DA:00:00:01 --device-id <device-id> --state-dir instances/main
```

The unit records the interpreter that runs `create`, so use the venv's python: plain
`sudo python3` writes `/usr/bin/python3` into `ExecStart` and the unit fails on the first
missing import.

Then put the AI settings in the per-instance env file the unit reads:

```bash
sudo tee /path/to/diy-key/diykey/instances/main/diykey.env >/dev/null <<'EOF'
# COCO ONNX detector (Ultralytics anchor-free export, e.g. pi-port's yolov9-s-320.onnx).
DIYKEY_RAM_MODEL=/root/models/yolov9-s-320.onnx
# Find Anything (CLIP ViT-L/14 + the nl-search WebSocket). Optional, heavy.
# DIYKEY_CLIP=1
# DIYKEY_NL_SEARCH=1
# DIYKEY_RAM_TAGS=/root/up/app/fixtures/ram_tags.csv
# DIYKEY_FW_DIR=/root/diy-key-firmware
# Features advertised to Protect are on by default, like the firmware. Set a key to 0/off to
# hide that capability; a missing model fails its task, not the device.
# DIYKEY_FACE=0          # hide face recognition
# DIYKEY_FACE_MODELS=/path/to/models
# DIYKEY_LPR=0           # hide licence plate recognition
# DIYKEY_FACE_ENHANCE=0  # hide face enhancement
# DIYKEY_FACE_ENHANCE_MODEL=/path/to/gfpgan_1.4.onnx
# DIYKEY_VLM=transformers  # AI summary on Linux (mlx is macOS only); needs pip install transformers
# DIYKEY_VLM_MODEL=/path/to/InternVL2.5-1B   # local checkpoint for transformers
# DIYKEY_TASK_WORKERS=1
# DIYKEY_TASK_QUEUE_MAX=200
EOF
sudo systemctl restart diy-key-instance-main
```

`EnvironmentFile` is `-`-prefixed, so a missing file does not stop the unit. Identity and
network stay in the unit (macvlan MAC, `--state-dir`); the env file is for the AI settings
only. `DIYKEY_FW_VERSION` / `DIYKEY_DISCOVERY_FW_VERSION` also work there if you want to pin a
factory version without touching `diykey.cfg`. The AI settings keys are in
`scripts/diykey.env.example`; identity and network overrides are in `core/config.py`.

## Verify

```bash
journalctl -u diy-key-instance-main -f
```

Expected once adopted: `WSS CONNECTED`, and with the detector configured, `RAM tagger: …` at
the first `recognizeKeyFrames` task followed by `event …: <description>` and
`POST …/recognize-anything -> 200`. With CLIP, `NL_PARSE … -> tags … embed 768` and
`nl-search connected to wss://<console>:7443/wss/nl-search/v1`.

## Known limits on this runtime

- **No NPU path for the new stages.** `stage5_faceid/` still uses `rknnlite`, but RAM tagging,
  CLIP and verification run on CPU via onnxruntime/torch. Feeding the detector through RKNN is
  not implemented.
- **One shared PostgreSQL.** The console's result database is the host cluster, so several
  instances on one board would share `:5432`; the macOS per-instance cluster is not used.
- **RAM verified, CLIP not.** RAM tagging ran live on the board (2026-10-30, CPU); CLIP and
  Find Anything have only been exercised on macOS against Protect 7.3.70.
- `scripts/diykey_*.sh` remain the macOS service; the Linux path is systemd only.
