# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real RK3588 hardware stats for the `systemMetrics` ucp4 RPC.

The reply is a flat array of {name, value} fractions. That shape and scale come from the
real key's 2.2.8 firmware: ``uaibd/hwmon/aikey_monitor.py`` samples every second, averages
over 30 s for the first five minutes and over 300 s afterwards, then writes
CPU/GPU/Memory/PsqlUsage/curTops to ``/var/log/uaibd.hwmon.systemMetrics.log``;
``ui-websocketd`` serves that cached file when Protect asks for ``systemMetrics``. We run
the same kind of sampler so a Protect 30-minute bucket holds an average of the period
instead of whichever instant the 5-minute poll landed on.

The scale matters: Protect's UsageChart plots ``100*value`` against a 0-100 axis
(``@ubnt/unifi-protect-ui-internal`` vantage.js, ``y:100*e.value``). Sending percentages
pins CPU and Memory at the top of the graph, which is what the old pre-restructure build
did. Internally CPU/GPU are 0-100 and become 0-1 in the RPC; Memory is already 0-1
used/total.
"""
import glob
import logging
import os
import threading
import time

log = logging.getLogger("system-stats")

SAMPLE_INTERVAL_S = 1.0
SHORT_BLOCK_S = 30.0   # firmware's first 5 minutes
LONG_BLOCK_S = 300.0   # then a 5-minute average, one Protect poll
SHORT_PHASE_S = 300.0

THERMAL_ZONE_TYPES = {}


def _load_thermal_zone_types():
    global THERMAL_ZONE_TYPES
    if THERMAL_ZONE_TYPES:
        return THERMAL_ZONE_TYPES
    zones = {}
    for type_path in glob.glob("/sys/class/thermal/thermal_zone*/type"):
        try:
            with open(type_path) as f:
                name = f.read().strip()
            zone_dir = os.path.dirname(type_path)
            zones[name] = os.path.join(zone_dir, "temp")
        except OSError:
            continue
    THERMAL_ZONE_TYPES = zones
    return zones


def _read_temp_c(zone_type: str):
    zones = _load_thermal_zone_types()
    path = zones.get(zone_type)
    if not path:
        return None
    try:
        with open(path) as f:
            millidegrees = int(f.read().strip())
        return round(millidegrees / 1000.0, 1)
    except (OSError, ValueError):
        return None


def _read_cpu_times():
    """(idle jiffies, total jiffies) from the aggregate /proc/stat line."""
    with open("/proc/stat") as f:
        parts = f.readline().split()
    vals = [int(x) for x in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return idle, sum(vals)


def _cpu_load_percent(previous, current):
    """CPU busy share between two /proc/stat reads, or None when there is no delta."""
    if previous is None or current is None:
        return None
    d_idle = current[0] - previous[0]
    d_total = current[1] - previous[1]
    if d_total <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - d_idle / d_total)))


def _read_cpu_load_percent(sample_interval=0.15):
    """One CPU load % sample, for the standalone CLI and diagnostics."""
    try:
        first = _read_cpu_times()
        time.sleep(sample_interval)
        load = _cpu_load_percent(first, _read_cpu_times())
    except (OSError, ValueError, IndexError):
        return None
    return round(load, 1) if load is not None else None


def _read_ram_kb():
    total_kb = avail_kb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
    except (OSError, ValueError):
        pass
    return total_kb, avail_kb


def _ram_used_fraction():
    total_kb, avail_kb = _read_ram_kb()
    if not total_kb or avail_kb is None:
        return None
    return max(0.0, min(1.0, (total_kb - avail_kb) / total_kb))


def _read_storage_bytes(path="/"):
    try:
        st = os.statvfs(path)
        total = st.f_frsize * st.f_blocks
        free = st.f_frsize * st.f_bavail
        return total, total - free
    except OSError:
        return None, None


def _read_gpu_load_percent():
    # The platform node is a plain 0-100 integer; the devfreq one is "12@300000000Hz".
    # Prefer the plain node and fall back to devfreq for kernels that only expose that.
    try:
        with open("/sys/devices/platform/fb000000.gpu/utilisation") as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        pass
    for path in glob.glob("/sys/class/devfreq/*.gpu/load"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            return float(raw.split("@")[0])
        except (OSError, ValueError, IndexError):
            continue
    return None


def _read_npu_load_percent():
    path = "/sys/kernel/debug/rknpu/load"
    try:
        with open(path) as f:
            raw = f.read().strip()  # "NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,"
    except OSError:
        return None
    loads = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if "%" in chunk and ":" in chunk:
            try:
                loads.append(float(chunk.split(":")[-1].replace("%", "").strip()))
            except ValueError:
                continue
    if not loads:
        return None
    return round(sum(loads) / len(loads), 1)


class _MetricsSampler:
    """Firmware-shaped rolling average: one sample a second, block averages on flush."""

    def __init__(self, interval=SAMPLE_INTERVAL_S, short_block=SHORT_BLOCK_S,
                 long_block=LONG_BLOCK_S, short_phase=SHORT_PHASE_S,
                 cpu_times=_read_cpu_times, gpu_load=_read_gpu_load_percent,
                 ram_used=_ram_used_fraction, clock=time.monotonic):
        self.interval = interval
        self.short_block = short_block
        self.long_block = long_block
        self.short_phase = short_phase
        self.cpu_times = cpu_times
        self.gpu_load = gpu_load
        self.ram_used = ram_used
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._values = {"cpu": None, "gpu": None, "memory": None}

    def start(self):
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, daemon=True, name="diykey-system-metrics")
                self._thread.start()

    def snapshot(self):
        with self._lock:
            return dict(self._values)

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2)

    def publish(self, samples):
        """Publish one block average; an empty metric keeps its previous value."""
        with self._lock:
            for name, values in samples.items():
                if values:
                    self._values[name] = sum(values) / len(values)

    def _run(self):
        started = self._clock()
        block_end = started + self.short_block
        previous = None
        samples = {"cpu": [], "gpu": [], "memory": []}
        failing = False
        while not self._stop.is_set():
            try:
                try:
                    current = self.cpu_times()
                except (OSError, ValueError, IndexError):
                    current = None
                cpu = _cpu_load_percent(previous, current)
                previous = current
                if cpu is not None:
                    samples["cpu"].append(cpu)
                gpu = self.gpu_load()
                if gpu is not None:
                    samples["gpu"].append(gpu)
                memory = self.ram_used()
                if memory is not None:
                    samples["memory"].append(memory)
                failing = False
            except Exception:
                # Keep the last published average rather than freezing or dying silently.
                if not failing:
                    log.exception("system metrics sample failed; keeping the last average")
                    failing = True

            now = self._clock()
            if now >= block_end:
                self.publish(samples)
                samples = {"cpu": [], "gpu": [], "memory": []}
                block_end = now + (
                    self.short_block if now - started < self.short_phase else self.long_block)
            self._stop.wait(self.interval)


_sampler = _MetricsSampler()


def start_system_metrics_sampler():
    _sampler.start()


def _metrics(cpu_load, gpu_load, ram_used):
    total_kb, avail_kb = _read_ram_kb()
    if ram_used is None and total_kb and avail_kb is not None:
        ram_used = max(0.0, min(1.0, (total_kb - avail_kb) / total_kb))
    storage_total, storage_used = _read_storage_bytes()
    return {
        "cpu": {"load": cpu_load, "temperature": _read_temp_c("soc-thermal")},
        "gpu": {"load": gpu_load, "temperature": _read_temp_c("gpu-thermal")},
        # Kept for diagnostics; the real key reports no NPU metric over ucp4.
        "npu": {"load": _read_npu_load_percent(), "temperature": _read_temp_c("npu-thermal")},
        "ram": {"totalKilobytes": total_kb, "availableKilobytes": avail_kb,
                "usedFraction": ram_used},
        "storage": {"totalBytes": storage_total, "usedBytes": storage_used},
    }


def collect_system_metrics() -> dict:
    """Cached block averages, starting the sampler on first use."""
    _sampler.start()
    values = _sampler.snapshot()
    return _metrics(values["cpu"], values["gpu"], values["memory"])


def collect_instant_metrics() -> dict:
    """One 150 ms snapshot, for the CLI and diagnostics only."""
    return _metrics(_read_cpu_load_percent(), _read_gpu_load_percent(),
                    _ram_used_fraction())


if __name__ == "__main__":
    import json
    print(json.dumps(collect_instant_metrics(), indent=2))
