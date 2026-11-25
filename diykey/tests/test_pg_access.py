# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pg_access  # noqa: E402

# What scripts/diykey_pg_init.sh appends, for the console it was set up with.
INIT_BLOCK = ("\n# DIY-Key: only the Protect console, only over SSL.\n"
              "hostssl unifi-protect unifi-protect 192.168.1.1/32 scram-sha-256\n")
STOCK = ("# TYPE  DATABASE        USER            ADDRESS                 METHOD\n"
         "local   all             all                                     trust\n"
         "host    all             all             127.0.0.1/32            scram-sha-256\n")


class TestRenderHba(unittest.TestCase):
    def test_moves_the_init_rule_to_the_new_console(self):
        out = pg_access.render_hba(STOCK + INIT_BLOCK, "192.168.1.15")
        self.assertNotIn("192.168.1.1/32", out)
        self.assertTrue(out.endswith(
            pg_access.MARKER + "\n"
            "hostssl unifi-protect unifi-protect 192.168.1.15/32 scram-sha-256\n"))
        self.assertEqual(out.count(pg_access.MARKER), 1)
        self.assertIn("host    all             all             127.0.0.1/32", out)

    def test_idempotent(self):
        once = pg_access.render_hba(STOCK + INIT_BLOCK, "192.168.1.15")
        self.assertEqual(pg_access.render_hba(once, "192.168.1.15"), once)

    def test_adds_a_block_to_a_file_without_one(self):
        out = pg_access.render_hba(STOCK, "10.0.0.2")
        self.assertTrue(out.startswith(STOCK))
        self.assertIn("hostssl unifi-protect unifi-protect 10.0.0.2/32 scram-sha-256", out)

    def test_keeps_a_users_own_lan_rule(self):
        lan = "hostssl unifi-protect unifi-protect 192.168.1.0/24 scram-sha-256\n"
        out = pg_access.render_hba(STOCK + lan + INIT_BLOCK, "192.168.1.15")
        self.assertIn(lan, out)
        self.assertNotIn("192.168.1.1/32", out)

    def test_ipv6(self):
        self.assertIn("fd00::5/128", pg_access.render_hba(STOCK, "fd00::5"))


class TestConsoleAddress(unittest.TestCase):
    def test_literal(self):
        self.assertEqual(pg_access.console_address("192.168.1.15"), "192.168.1.15")

    def test_name_is_resolved(self):
        with mock.patch.object(pg_access.socket, "gethostbyname", return_value="10.1.2.3"):
            self.assertEqual(pg_access.console_address("protect.lan"), "10.1.2.3")

    def test_unresolvable(self):
        with mock.patch.object(pg_access.socket, "gethostbyname", side_effect=OSError):
            self.assertIsNone(pg_access.console_address("nowhere.invalid"))
        self.assertIsNone(pg_access.console_address(""))


class TestAdmits(unittest.TestCase):
    def _admits(self, rows, address="192.168.1.15"):
        with mock.patch.object(pg_access, "_query", return_value=rows):
            return pg_access.admits(address)

    def test_exact_rule(self):
        self.assertTrue(self._admits([["unifi-protect", "unifi-protect", "192.168.1.15",
                                       "255.255.255.255"]]))

    def test_old_console_only(self):
        self.assertFalse(self._admits([["unifi-protect", "unifi-protect", "192.168.1.1",
                                        "255.255.255.255"]]))

    def test_lan_rule_or_all(self):
        self.assertTrue(self._admits([["all", "all", "192.168.1.0", "255.255.255.0"]]))
        self.assertTrue(self._admits([["unifi-protect", "all", "all", ""]]))

    def test_other_database(self):
        self.assertFalse(self._admits([["other", "unifi-protect", "192.168.1.15",
                                        "255.255.255.255"]]))


class TestWritePreserving(unittest.TestCase):
    def test_keeps_mode_and_replaces_content(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        path = os.path.join(d, "pg_hba.conf")
        with open(path, "w") as f:
            f.write("old\n")
        os.chmod(path, 0o600)
        pg_access._write_preserving(path, "new\n")
        with open(path) as f:
            self.assertEqual(f.read(), "new\n")
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(os.listdir(d), ["pg_hba.conf"])  # no temp file left


class TestEnsureConsole(unittest.TestCase):
    """The flow with PostgreSQL mocked: what gets synced, reloaded and reported."""

    def _run(self, *, changed=True, errors=(), reloaded=True, admitted=True, listening=True,
             env=None, host="192.168.1.15"):
        calls = []
        patches = [
            mock.patch.object(pg_access, "sync_hba",
                              side_effect=lambda a: calls.append(("sync", a)) or changed),
            mock.patch.object(pg_access, "hba_errors", return_value=list(errors)),
            mock.patch.object(pg_access, "reload",
                              side_effect=lambda: calls.append("reload") or reloaded),
            mock.patch.object(pg_access, "admits", return_value=admitted),
            mock.patch.object(pg_access, "protect_sessions", return_value=4),
            mock.patch.object(pg_access, "listening", return_value=listening),
            mock.patch.dict(os.environ, env or {}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        ok = pg_access.ensure_console(host, bind_ip="192.168.1.18", session_check_delay=0)
        return ok, calls

    def test_new_console_is_synced_and_reloaded(self):
        with self.assertLogs("pg-access", level="INFO") as logs:
            ok, calls = self._run()
        self.assertTrue(ok)
        self.assertEqual(calls, [("sync", "192.168.1.15"), "reload"])
        self.assertIn("rule updated and reloaded", "\n".join(logs.output))

    def test_unchanged_file_is_not_reloaded(self):
        ok, calls = self._run(changed=False)
        self.assertTrue(ok)
        self.assertEqual(calls, [("sync", "192.168.1.15")])

    def test_parse_errors_stop_the_reload(self):
        with self.assertLogs("pg-access", level="WARNING") as logs:
            ok, calls = self._run(errors=["line 9: invalid IP address"])
        self.assertFalse(ok)
        self.assertNotIn("reload", calls)
        self.assertIn("keeps its old rules", "\n".join(logs.output))

    def test_sync_off_still_verifies(self):
        with self.assertLogs("pg-access", level="WARNING") as logs:
            ok, calls = self._run(admitted=False, env={"DIYKEY_PG_HBA_SYNC": "0"})
        self.assertFalse(ok)
        self.assertEqual(calls, [])
        self.assertIn("no hostssl rule", "\n".join(logs.output))

    def test_not_listening(self):
        with self.assertLogs("pg-access", level="WARNING") as logs:
            ok, _ = self._run(listening=False)
        self.assertFalse(ok)
        self.assertIn("listen_addresses", "\n".join(logs.output))

    def test_no_postgres_on_this_host(self):
        with mock.patch.object(pg_access, "psql_cmd", return_value=(None, "no DIYKEY_PG_SOCKET")):
            with self.assertLogs("pg-access", level="INFO") as logs:
                self.assertFalse(pg_access.ensure_console("192.168.1.15"))
        self.assertIn("not checked", "\n".join(logs.output))


class TestSessionCheck(unittest.TestCase):
    def setUp(self):
        self.gen = pg_access._generation

    def test_session_seen(self):
        with mock.patch.object(pg_access, "protect_sessions", return_value=5):
            with self.assertLogs("pg-access", level="INFO") as logs:
                pg_access._check_session_later(self.gen, "192.168.1.15", 4, 0)
        self.assertIn("opened 1 session", "\n".join(logs.output))

    def test_no_session_quotes_the_refusals(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        log_path = os.path.join(d, "postgres.log")
        with open(log_path, "w") as f:
            f.write('2026-11-25 FATAL:  no pg_hba.conf entry for host "192.168.1.15", user '
                    '"unifi-protect", database "unifi-protect", SSL encryption\n')
        with mock.patch.object(pg_access, "protect_sessions", return_value=4), \
                mock.patch.dict(os.environ, {"DIYKEY_PG_LOG": log_path}):
            with self.assertLogs("pg-access", level="WARNING") as logs:
                pg_access._check_session_later(self.gen, "192.168.1.15", 4, 0)
        out = "\n".join(logs.output)
        self.assertIn("no session from Protect", out)
        self.assertIn('no pg_hba.conf entry for host "192.168.1.15"', out)

    def test_superseded_check_stays_quiet(self):
        with mock.patch.object(pg_access, "protect_sessions") as sessions:
            pg_access._check_session_later(self.gen - 1, "192.168.1.15", 4, 0)
        sessions.assert_not_called()


def _pg_bin():
    for d in (os.environ.get("DIYKEY_PG_BIN"), "/opt/homebrew/opt/postgresql@17/bin"):
        if d and os.path.exists(os.path.join(d, "initdb")):
            return d
    found = shutil.which("initdb")
    return os.path.dirname(found) if found else None


@unittest.skipUnless(_pg_bin() and sys.platform == "darwin" and os.geteuid() != 0,
                     "needs a local PostgreSQL (initdb) on macOS, not as root")
class TestAgainstRealPostgres(unittest.TestCase):
    """A throwaway cluster set up like diykey_pg_init.sh (old console .1), moved to .15."""

    @classmethod
    def setUpClass(cls):
        cls.bin = _pg_bin()
        cls.dir = tempfile.mkdtemp(prefix="diykey-pg-")
        cls.data = os.path.join(cls.dir, "pg")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]
        # Like diykey_service.sh: without a valid LC_ALL the macOS postmaster refuses to start
        # ("postmaster became multithreaded during startup").
        env = {**os.environ, "LC_ALL": "en_US.UTF-8"}
        run = lambda *a: subprocess.run(a, check=True, capture_output=True, env=env)  # noqa: E731
        run(os.path.join(cls.bin, "initdb"), "-D", cls.data, "--auth-local=trust",
            "--auth-host=scram-sha-256", "-E", "UTF8", "--locale=en_US.UTF-8")
        run("openssl", "req", "-new", "-x509", "-days", "1", "-nodes", "-subj", "/CN=diy-key-test",
            "-keyout", os.path.join(cls.data, "server.key"),
            "-out", os.path.join(cls.data, "server.crt"))
        os.chmod(os.path.join(cls.data, "server.key"), 0o600)
        with open(os.path.join(cls.data, "postgresql.conf"), "a") as f:
            f.write(f"\nlisten_addresses = '127.0.0.1'\nport = {cls.port}\n"
                    f"unix_socket_directories = '{cls.data}'\nssl = on\n")
        with open(os.path.join(cls.data, "pg_hba.conf"), "a") as f:
            f.write(INIT_BLOCK)
        run(os.path.join(cls.bin, "pg_ctl"), "-D", cls.data, "-l",
            os.path.join(cls.dir, "postgres.log"), "-w", "start")
        run(os.path.join(cls.bin, "psql"), "-h", cls.data, "-p", str(cls.port), "-d", "postgres",
            "-qc", "CREATE DATABASE \"unifi-protect\";")

    @classmethod
    def tearDownClass(cls):
        subprocess.run([os.path.join(cls.bin, "pg_ctl"), "-D", cls.data, "-m", "immediate",
                        "-w", "stop"], capture_output=True)
        shutil.rmtree(cls.dir, ignore_errors=True)

    def setUp(self):
        env = {"DIYKEY_PG_SOCKET": self.data, "DIYKEY_PG_BIN": self.bin,
               "PGPORT": str(self.port), "DIYKEY_PG_LOG": os.path.join(self.dir, "postgres.log")}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(pg_access, "PG_PORT", self.port)
        p.start()
        self.addCleanup(p.stop)

    def test_moves_the_rule_reloads_and_verifies(self):
        self.assertFalse(pg_access.admits("192.168.1.15"))
        before = pg_access._conf_load_time()
        self.assertTrue(pg_access.ensure_console("192.168.1.15", bind_ip="127.0.0.1",
                                                 session_check_delay=0))
        self.assertTrue(pg_access.admits("192.168.1.15"))
        self.assertFalse(pg_access.admits("192.168.1.1"))
        self.assertNotEqual(pg_access._conf_load_time(), before)
        with open(os.path.join(self.data, "pg_hba.conf")) as f:
            self.assertEqual(f.read().count(pg_access.MARKER), 1)
        # Second connect to the same console: nothing to change.
        with mock.patch.object(pg_access, "reload") as reload:
            self.assertTrue(pg_access.ensure_console("192.168.1.15", bind_ip="127.0.0.1",
                                                     session_check_delay=0))
        reload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
