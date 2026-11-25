#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Run one DIY-Key instance on macOS from ~/.diy-key/<name>/ (see scripts/diykey_paths.sh).
# Touch restart to reload settings; Ctrl-C stops it and its PostgreSQL.
set -u

USER_HOME=$(eval echo "~${SUDO_USER:-${USER:-$(id -un)}}")
REPO=$(cd "$(dirname "$0")/.." && pwd)
# ENV_FILE, TRIGGER, LOG, PIDFILE, DEFAULT_STATE_DIR, PG_DIR and PG_LOG.
. "$(dirname "$0")/diykey_paths.sh"

# Under root (scripts/macos_launchd.sh) the instance adds its own address: an alias isn't kept
# across reboots, and this runs at boot.
ensure_alias() {
    [ -n "${DIYKEY_BIND_IP:-}" ] && [ "$(id -u)" -eq 0 ] || return 0
    ifconfig "$DIYKEY_IFACE" 2>/dev/null | grep -q "inet $DIYKEY_BIND_IP " && return 0
    echo "diykey_service: adding $DIYKEY_BIND_IP to $DIYKEY_IFACE"
    ifconfig "$DIYKEY_IFACE" alias "$DIYKEY_BIND_IP" 255.255.255.255
}

load_settings() {
    # Clear the previous run's settings so a removed line really goes away.
    for k in $(env | grep -E '^DIYKEY_[A-Z0-9_]*=' | cut -d= -f1); do
        case $k in
            DIYKEY_INSTANCE|DIYKEY_ENV_FILE|DIYKEY_RESTART_FILE|DIYKEY_LOG|DIYKEY_PIDFILE) ;;
            *) unset "$k" ;;
        esac
    done
    if [ ! -r "$ENV_FILE" ]; then
        echo "diykey_service: no $ENV_FILE (copy scripts/diykey.env.example)" >&2
        return 1
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%%#*}
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [ -z "$line" ] && continue
        key=${line%%=*}
        value=${line#*=}
        value=${value#\"}; value=${value%\"}
        if [[ "$key" =~ ^DIYKEY_[A-Z0-9_]+$ && "$line" == *=* ]]; then
            export "$key=$value"
        else
            echo "diykey_service: ignoring line in $ENV_FILE: $line" >&2
        fi
    done < "$ENV_FILE"
    export DIYKEY_STATE_DIR=${DIYKEY_STATE_DIR:-$DEFAULT_STATE_DIR}
    local missing=""
    for k in DIYKEY_IFACE DIYKEY_MAC DIYKEY_DEVICE_ID DIYKEY_GUID; do
        [ -n "${!k:-}" ] || missing="$missing $k"
    done
    if [ -n "$missing" ]; then
        echo "diykey_service: $ENV_FILE is missing$missing" >&2
        return 1
    fi
    ensure_alias
    if [ -n "${DIYKEY_BIND_IP:-}" ] && ! ifconfig "$DIYKEY_IFACE" 2>/dev/null | grep -q "inet $DIYKEY_BIND_IP "; then
        echo "diykey_service: $DIYKEY_BIND_IP is not on $DIYKEY_IFACE; add it with" \
             "sudo ifconfig $DIYKEY_IFACE alias $DIYKEY_BIND_IP 255.255.255.255" >&2
        return 1
    fi
}

# PostgreSQL refuses to run as root: under the daemon it runs as the owner of its cluster.
pg_ctl() {
    local default_bin=/opt/homebrew/opt/postgresql@17/bin
    local bin=${DIYKEY_PG_BIN:-$default_bin}/pg_ctl
    if [ "$(id -u)" -eq 0 ] && [ -n "${DIYKEY_PG_BIN:-}" ] && [ "$DIYKEY_PG_BIN" != "$default_bin" ]; then
        # A user-writable env file must not point a root daemon at arbitrary code.
        local owner mode
        owner=$(stat -f %Su "$bin" 2>/dev/null || echo missing)
        mode=$(stat -f %Lp "$bin" 2>/dev/null || echo 777)
        if [ "$owner" != root ] || [ $((8#$mode & 8#022)) -ne 0 ]; then
            echo "diykey_service: refusing root-executed DIYKEY_PG_BIN=$bin (owner=$owner mode=$mode)" >&2
            return 1
        fi
    fi
    if [ "$(id -u)" -eq 0 ]; then
        sudo -u "$(stat -f %Su "$PG_DIR")" env LC_ALL=en_US.UTF-8 "$bin" "$@"
    else
        "$bin" "$@"
    fi
}
pg_wanted() { [ "${DIYKEY_PG:-1}" != 0 ] && [ -f "$PG_DIR/PG_VERSION" ]; }
ensure_pg() {
    pg_wanted || return 0
    pg_ctl -D "$PG_DIR" status >/dev/null 2>&1 && return 0
    echo "diykey_service: starting PostgreSQL in $PG_DIR (log $PG_LOG)"
    LC_ALL=en_US.UTF-8 pg_ctl -D "$PG_DIR" -l "$PG_LOG" -w start >/dev/null ||
        echo "diykey_service: PostgreSQL did not start; see $PG_LOG" >&2
}
stop_pg() {
    pg_wanted && pg_ctl -D "$PG_DIR" status >/dev/null 2>&1 || return 0
    echo "diykey_service: stopping PostgreSQL"
    pg_ctl -D "$PG_DIR" -m fast -w stop >/dev/null
}

mtime() { stat -f %m "$1" 2>/dev/null || echo 0; }

child=""
stop_child() {
    if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
        kill -INT -- "-$child" 2>/dev/null || kill -INT "$child"   # its whole process group
        kill -CONT -- "-$child" 2>/dev/null   # a stopped process can't act on the signal
        for _ in $(seq 1 20); do kill -0 "$child" 2>/dev/null || break; sleep 0.5; done
        kill -TERM -- "-$child" 2>/dev/null
        kill -CONT -- "-$child" 2>/dev/null
        wait "$child" 2>/dev/null
    fi
    child=""
}
stop_stale() {
    # A DIY-Key left by an earlier wrapper that was killed outright: it has its own process
    # group, so it outlives the wrapper and keeps :8080.
    local old
    old=$(cat "$PIDFILE" 2>/dev/null) || return 0
    # This instance's only: as root a reused pid could be another service's run_all.py.
    if [ -n "$old" ] && ps -p "$old" -o command= 2>/dev/null | grep "run_all.py" \
            | grep -qF -- "--state-dir $DIYKEY_STATE_DIR"; then
        echo "diykey_service: stopping DIY-Key (pid $old) left running by an earlier start"
        child=$old
        stop_child
    fi
    rm -f "$PIDFILE"
}
trap 'echo "diykey_service: stopping"; stop_child; stop_pg; rm -f "$PIDFILE"; exit 0' INT TERM

# One writer rotates the log; it ignores Ctrl-C/TERM so the stop is still logged. Its sizes are
# read once, here (a change needs a full restart, not the restart file).
env_number() {
    local v
    v=$(sed -n "s/^[[:space:]]*$1=//p" "$ENV_FILE" 2>/dev/null | tail -1 | sed 's/[[:space:]#].*//')
    [[ "$v" =~ ^[0-9]+(\.[0-9]+)?$ ]] && echo "$v" || echo "$2"
}
mkdir -p "$(dirname "$LOG")"
exec > >("$REPO/venv/bin/python" "$(dirname "$0")/log_writer.py" "$LOG" \
             --max-mb "$(env_number DIYKEY_LOG_MAX_MB 50)" \
             --keep "$(env_number DIYKEY_LOG_KEEP 5)") 2>&1
caffeinate -is -w $$ &

while true; do
    if ! load_settings; then
        sleep 5
        continue
    fi
    ensure_pg
    # changeUserPassword applies Protect's rotated password to this cluster's role, and each
    # ucp4 connect admits that console in pg_hba.conf and checks Protect gets in (pg_access.py;
    # it quotes refusals from the log). psql connects as the cluster's owner: under the daemon
    # there's no "root" role.
    if pg_wanted; then
        export DIYKEY_PG_SOCKET=$PG_DIR DIYKEY_PG_LOG=$PG_LOG
        DIYKEY_PG_USER=$(stat -f %Su "$PG_DIR")
        export DIYKEY_PG_USER
    else
        unset DIYKEY_PG_SOCKET DIYKEY_PG_USER DIYKEY_PG_LOG
    fi
    stop_stale
    if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
        # Root-created state and logs must not block a later run as the user.
        chown -R "$SUDO_USER":staff "$BASE/state" "$BASE/logs" \
            "$USER_HOME/.diy-key/models" 2>/dev/null || true
    fi
    seen=$(mtime "$TRIGGER")
    echo "diykey_service: starting DIY-Key '$INSTANCE' at $(date '+%F %T') (settings from $ENV_FILE)"
    bind=()
    [ -n "${DIYKEY_BIND_IP:-}" ] && bind=(--bind-ip "$DIYKEY_BIND_IP")
    # Job control only for this launch, so DIY-Key gets its own process group (stop_child
    # signals all of it); SIGINT is restored before exec so run_all can shut down cleanly.
    set -m
    (cd "$REPO/diykey" && exec "$REPO/venv/bin/python" -c 'import os, signal, sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
os.execv(sys.argv[1], sys.argv[1:])' "$REPO/venv/bin/python" run_all.py \
        --iface "$DIYKEY_IFACE" --state-dir "$DIYKEY_STATE_DIR" --mac "$DIYKEY_MAC" \
        --device-id "$DIYKEY_DEVICE_ID" --skip-grpc ${bind[@]+"${bind[@]}"}) < /dev/null &
    child=$!
    echo "$child" > "$PIDFILE"
    set +m
    while kill -0 "$child" 2>/dev/null; do
        sleep 1
        if [ "$(mtime "$TRIGGER")" != "$seen" ]; then
            echo "diykey_service: $TRIGGER touched, restarting"
            stop_child
            break
        fi
    done
    if [ -n "$child" ]; then
        wait "$child"
        echo "diykey_service: DIY-Key exited ($?), restarting in 5 s"
        child=""
        sleep 5
    fi
done
