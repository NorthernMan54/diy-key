#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Create an instance's PostgreSQL for Protect's AI results (Homebrew postgresql@17 + pgvector).
# Reads DIYKEY_BIND_IP and DIYKEY_CONSOLE_HOST; diykey_service.sh starts and stops it.
set -eu

USER_HOME=$(eval echo "~${SUDO_USER:-${USER:-$(id -un)}}")
. "$(dirname "$0")/diykey_paths.sh"

setting() { sed -n "s/^[[:space:]]*$1=//p" "$ENV_FILE" | tail -1 | sed 's/[[:space:]#].*//'; }
[ -r "$ENV_FILE" ] || { echo "no $ENV_FILE (copy scripts/diykey.env.example)" >&2; exit 1; }
BIND_IP=$(setting DIYKEY_BIND_IP)
CONSOLE=$(setting DIYKEY_CONSOLE_HOST)
PG_BIN=$(setting DIYKEY_PG_BIN); PG_BIN=${PG_BIN:-/opt/homebrew/opt/postgresql@17/bin}
[ -n "$BIND_IP" ] && [ -n "$CONSOLE" ] || { echo "$ENV_FILE needs DIYKEY_BIND_IP and DIYKEY_CONSOLE_HOST" >&2; exit 1; }
[ -e "$PG_DIR" ] && { echo "$PG_DIR already exists" >&2; exit 1; }
[ -x "$PG_BIN/initdb" ] || { echo "no $PG_BIN/initdb: brew install postgresql@17 pgvector" >&2; exit 1; }

mkdir -p "$(dirname "$PG_LOG")"
"$PG_BIN/initdb" -D "$PG_DIR" -U "$(id -un)" --auth-local=trust --auth-host=scram-sha-256 \
    -E UTF8 --locale=en_US.UTF-8 >/dev/null
openssl req -new -x509 -days 3650 -nodes -subj "/CN=diy-key-$INSTANCE" \
    -keyout "$PG_DIR/server.key" -out "$PG_DIR/server.crt" 2>/dev/null
chmod 600 "$PG_DIR/server.key"
cat >> "$PG_DIR/postgresql.conf" <<EOF

# DIY-Key '$INSTANCE': the AI Key's results DB for Protect, on the instance's own address only.
listen_addresses = '$BIND_IP'
port = 5432
unix_socket_directories = '$PG_DIR'
ssl = on
password_encryption = scram-sha-256
EOF
# pg_access.py rewrites the line after this marker on each ucp4 connect, for the console being
# connected to: keep the marker text in sync with pg_access.MARKER.
cat >> "$PG_DIR/pg_hba.conf" <<EOF

# DIY-Key: only the Protect console, only over SSL.
hostssl unifi-protect unifi-protect $CONSOLE/32 scram-sha-256
EOF

LC_ALL=en_US.UTF-8 "$PG_BIN/pg_ctl" -D "$PG_DIR" -l "$PG_LOG" -w start >/dev/null
trap '"$PG_BIN/pg_ctl" -D "$PG_DIR" -m fast -w stop >/dev/null' EXIT
# Superuser: Protect's migrations create extensions (vector, pg_trgm, intarray).
"$PG_BIN/psql" -h "$PG_DIR" -d postgres -v ON_ERROR_STOP=1 -q \
    -c "CREATE ROLE \"unifi-protect\" WITH LOGIN SUPERUSER PASSWORD 'unifi-protect';" \
    -c "CREATE DATABASE \"unifi-protect\" OWNER \"unifi-protect\";"
"$PG_BIN/psql" -h "$PG_DIR" -d unifi-protect -v ON_ERROR_STOP=1 -q -c "CREATE EXTENSION vector;"
echo "created $PG_DIR (listens on $BIND_IP:5432, admits $CONSOLE); diykey_service.sh starts it"
