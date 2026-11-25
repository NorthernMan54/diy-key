# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep the instance's PostgreSQL open to the console that adopted it, and check it works.

Protect opens the AI Key's PostgreSQL as user ``unifi-protect`` once the ucp4 socket is up,
and makes the key its search host only after that succeeds. Until then it keeps the key's
results out of search and hides the dashboard's Edge AI Latency graph. ``diykey_pg_init.sh``
writes a single ``hostssl`` rule for ``DIYKEY_CONSOLE_HOST`` when the cluster is created. A
console that moves or re-adopts the key from another address then gets
``no pg_hba.conf entry for host "<new console>"`` (Protect aiprocessors.db.log, 2026-11-25,
after Protect moved from 192.168.1.1 to 192.168.1.15), and the ucp4 side looks healthy.

Before each ucp4 connection, ``ensure_console`` rewrites DIY-Key's own rule (the marker line
and the ``hostssl unifi-protect unifi-protect <addr> scram-sha-256`` line right after it) for
the console being connected to and reloads PostgreSQL. It then checks that:

* the file parses (``pg_hba_file_rules.error``; PostgreSQL keeps its old rules on a bad file),
* the reload happened (``pg_conf_load_time()`` moved),
* PostgreSQL accepts TCP on the key's address, and
* a few minutes later, Protect has opened a session (``pg_stat_database.sessions`` for
  ``unifi-protect`` went up). If not, it logs the refusals from the PostgreSQL log.

Other rules in pg_hba.conf are left alone. ``DIYKEY_PG_HBA_SYNC=0`` keeps the file as is and
only runs the checks.
"""
import ipaddress
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

log = logging.getLogger("pg-access")

MARKER = "# DIY-Key: only the Protect console, only over SSL."
_RULE_RE = re.compile(r"^\s*hostssl\s+unifi-protect\s+unifi-protect\s+\S+\s+scram-sha-256\s*$")
PG_PORT = 5432
# Protect connects right after the ucp4 socket opens, then retries with backoff (10 s, 20 s,
# 40 s, 80 s: aiprocessors.db.log). Three minutes covers that first round.
SESSION_CHECK_DELAY = 180.0

_generation = 0
_generation_lock = threading.Lock()


class PgAccessError(Exception):
    pass


def psql_path():
    """psql from DIYKEY_PG_BIN, else PATH, else Homebrew's postgresql@17 (macOS service)."""
    pg_bin = os.environ.get("DIYKEY_PG_BIN")
    if pg_bin:
        return os.path.join(pg_bin, "psql")
    return shutil.which("psql") or "/opt/homebrew/opt/postgresql@17/bin/psql"


def psql_cmd(database="postgres"):
    """(argv, None) for a superuser psql on this host's PostgreSQL, or (None, reason).

    macOS service: DIYKEY_PG_SOCKET=<instance>/pg, psql runs as the login user.
    Linux system PostgreSQL: none needed -- as root psql runs as the postgres OS user over
    /var/run/postgresql, so peer auth makes it a superuser. DIYKEY_PG_USER overrides the OS
    user (any user that can authenticate as a superuser works).
    """
    sock = os.environ.get("DIYKEY_PG_SOCKET")
    user = os.environ.get("DIYKEY_PG_USER")
    if sys.platform != "darwin" and not sock and os.geteuid() == 0:
        sock = "/var/run/postgresql"
        user = user or "postgres"
    if not sock and not user:
        return None, "no DIYKEY_PG_SOCKET/DIYKEY_PG_USER"
    cmd = [psql_path(), "-d", database, "-q", "-X", "-v", "ON_ERROR_STOP=1"]
    if sock:
        cmd += ["-h", sock]
    runuser = shutil.which("runuser")
    if user and os.geteuid() == 0 and user != "root" and runuser:
        cmd = [runuser, "-u", user, "--"] + cmd
    elif user and user != "root":
        cmd += ["-U", user]
    return cmd, None


def _query(sql, timeout=10):
    """Rows of ``sql`` as lists of strings (unaligned, '|'-separated)."""
    cmd, reason = psql_cmd()
    if cmd is None:
        raise PgAccessError(reason)
    try:
        r = subprocess.run(cmd + ["-At", "-F", "|"], input=sql, capture_output=True,
                           text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        raise PgAccessError(f"psql failed: {e!r}") from e
    if r.returncode:
        raise PgAccessError(f"psql exited {r.returncode}: {r.stderr.strip()}")
    return [line.split("|") for line in r.stdout.splitlines() if line]


def console_address(host):
    """The console's IP as pg_hba wants it (a literal, no DNS at connect time), or None."""
    if not host:
        return None
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None


def rule_for(address):
    prefix = 128 if ipaddress.ip_address(address).version == 6 else 32
    return f"hostssl unifi-protect unifi-protect {address}/{prefix} scram-sha-256"


def render_hba(text, address):
    """``text`` with DIY-Key's block (the marker and the rule right after it) replaced by one
    rule for ``address``, at the end. Rules a user added -- a LAN-wide ``hostssl`` line, say --
    stay where they are."""
    kept, skip_rule = [], False
    for line in text.splitlines():
        if line.strip() == MARKER:
            skip_rule = True
            continue
        if skip_rule and _RULE_RE.match(line):
            skip_rule = False
            continue
        skip_rule = False
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept + ["", MARKER, rule_for(address)]) + "\n"


def _write_preserving(path, text):
    """Replace ``path`` atomically, keeping its owner and mode: under the root daemon the file
    still belongs to the cluster's owner, and PostgreSQL won't read one it can't open."""
    st = os.stat(path)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".pg_hba.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, st.st_mode & 0o7777)
        if os.geteuid() == 0:
            os.chown(tmp, st.st_uid, st.st_gid)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sync_hba(address):
    """Point DIY-Key's rule at ``address``. True if the file changed."""
    path = _query("SHOW hba_file;")[0][0]
    with open(path) as f:
        text = f.read()
    new = render_hba(text, address)
    if new == text:
        return False
    _write_preserving(path, new)
    log.info("pg_hba.conf: now admitting the console at %s (%s)", address, path)
    return True


def hba_errors():
    """Parse errors PostgreSQL would hit if it reloaded pg_hba.conf now."""
    return [f"line {n}: {err}" for n, err in
            _query("SELECT line_number, error FROM pg_hba_file_rules WHERE error IS NOT NULL;")]


def _conf_load_time():
    return _query("SELECT pg_conf_load_time();")[0][0]


def reload():
    """Reload PostgreSQL's config and wait for it to apply. True when the load time moved."""
    before = _conf_load_time()
    _query("SELECT pg_reload_conf();")
    for _ in range(20):
        if _conf_load_time() != before:
            return True
        time.sleep(0.25)
    return False


def admits(address):
    """True when the current pg_hba.conf has a hostssl unifi-protect rule covering ``address``."""
    ip = ipaddress.ip_address(address)
    for db, user, addr, mask in _query(
            "SELECT array_to_string(database, ','), array_to_string(user_name, ','), "
            "coalesce(address, ''), coalesce(netmask, '') FROM pg_hba_file_rules "
            "WHERE type = 'hostssl' AND error IS NULL;"):
        if not ({"unifi-protect", "all"} & set(db.split(","))
                and {"unifi-protect", "all"} & set(user.split(","))):
            continue
        if addr == "all":
            return True
        try:
            net = ipaddress.ip_network(f"{addr}/{mask}" if mask else addr, strict=False)
        except ValueError:
            continue  # hostnames and samehost/samenet: can't tell from here
        if ip in net:
            return True
    return False


def listening(bind_ip, port=None, timeout=2.0):
    try:
        with socket.create_connection((bind_ip, port or PG_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def protect_sessions():
    rows = _query("SELECT sessions FROM pg_stat_database WHERE datname = 'unifi-protect';")
    return int(rows[0][0]) if rows and rows[0][0] else 0


def _refusals(address, limit=3):
    """Recent ``no pg_hba.conf entry`` lines from the PostgreSQL log (DIYKEY_PG_LOG)."""
    path = os.environ.get("DIYKEY_PG_LOG")
    if not path:
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 256 * 1024))
            tail = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    hits = [line.strip() for line in tail if "no pg_hba.conf entry" in line]
    return hits[-limit:]


def ensure_console(host, bind_ip=None, session_check_delay=SESSION_CHECK_DELAY):
    """Admit the console at ``host``, reload, and verify (see the module docstring).

    Returns True when PostgreSQL is set up for it (the session check runs later, in its own
    thread). Never raises: a host without a reachable PostgreSQL just logs and returns False.
    """
    global _generation
    with _generation_lock:
        _generation += 1
        generation = _generation
    address = console_address(host)
    if not address:
        log.warning("PostgreSQL access: can't resolve console host %r; pg_hba.conf left as is",
                    host)
        return False
    try:
        changed = False
        if os.environ.get("DIYKEY_PG_HBA_SYNC", "1") != "0":
            changed = sync_hba(address)
        errors = hba_errors()
        if errors:
            log.warning("pg_hba.conf has errors, so PostgreSQL keeps its old rules: %s",
                        "; ".join(errors))
            return False
        if changed and not reload():
            log.warning("PostgreSQL didn't confirm the reload; the console at %s may still be "
                        "refused", address)
        if not admits(address):
            log.warning("pg_hba.conf has no hostssl rule for user unifi-protect from %s: "
                        "Protect can't open this key's database (DIYKEY_PG_HBA_SYNC=0?)",
                        address)
            return False
        baseline = protect_sessions()
    except (PgAccessError, OSError, IndexError, ValueError) as e:
        log.info("PostgreSQL access not checked: %s", e)
        return False
    if bind_ip and bind_ip not in ("0.0.0.0", "::") and not listening(bind_ip):
        log.warning("PostgreSQL isn't accepting connections on %s:%d; check listen_addresses "
                    "in postgresql.conf", bind_ip, PG_PORT)
        return False
    log.info("PostgreSQL access: console %s admitted%s", address,
             " (rule updated and reloaded)" if changed else "")
    if session_check_delay:
        threading.Thread(target=_check_session_later,
                         args=(generation, address, baseline, session_check_delay),
                         daemon=True, name="diykey-pg-check").start()
    return True


def _check_session_later(generation, address, baseline, delay):
    time.sleep(delay)
    if generation != _generation:
        return  # a newer connection has its own check
    try:
        now = protect_sessions()
    except PgAccessError as e:
        log.info("PostgreSQL session check skipped: %s", e)
        return
    if now > baseline:
        log.info("PostgreSQL: Protect opened %d session(s) since the console connected",
                 now - baseline)
        return
    refused = _refusals(address)
    log.warning("PostgreSQL: no session from Protect %.0f s after the console connected; Protect "
                "won't use this key as its search host.%s", delay,
                (" Recent refusals: " + " | ".join(refused)) if refused else
                " Check Protect's aiprocessors.db.log.")
