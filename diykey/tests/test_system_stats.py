# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Unit tests for system_stats.py: CPU delta maths and the firmware-shaped sampler."""
import os
import sys
import time
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import system_stats  # noqa: E402


class TestCpuDelta(unittest.TestCase):
    def test_busy_share(self):
        # 10 idle jiffies out of 20 total -> 50% busy.
        self.assertEqual(system_stats._cpu_load_percent((100, 200), (110, 220)), 50.0)

    def test_no_time_passed(self):
        self.assertIsNone(system_stats._cpu_load_percent((100, 200), (100, 200)))

    def test_first_sample_has_no_previous(self):
        self.assertIsNone(system_stats._cpu_load_percent(None, (100, 200)))


class TestSampler(unittest.TestCase):
    def test_publish_averages_each_metric(self):
        sampler = system_stats._MetricsSampler()
        sampler._values = {"cpu": 10.0, "gpu": 20.0, "memory": 0.5}
        sampler.publish({"cpu": [20.0, 40.0], "gpu": [], "memory": [0.4, 0.6]})
        self.assertEqual(sampler.snapshot(), {"cpu": 30.0, "gpu": 20.0, "memory": 0.5})

    def test_thread_publishes_block_average(self):
        state = {"n": 0}

        def cpu_times():
            state["n"] += 1
            # Every read adds 1 idle and 4 total jiffies, so each delta is 75% busy.
            return state["n"], state["n"] * 4

        sampler = system_stats._MetricsSampler(
            interval=0.002, short_block=0.02, long_block=0.02, short_phase=0.0,
            cpu_times=cpu_times, gpu_load=lambda: 40.0, ram_used=lambda: 0.5)
        self.addCleanup(sampler.stop)
        sampler.start()

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and sampler.snapshot()["cpu"] is None:
            time.sleep(0.005)
        values = sampler.snapshot()
        self.assertAlmostEqual(values["cpu"], 75.0, places=3)
        self.assertAlmostEqual(values["gpu"], 40.0, places=3)
        self.assertAlmostEqual(values["memory"], 0.5, places=3)


class TestCollect(unittest.TestCase):
    def test_uses_sampler_average_and_used_fraction(self):
        class FakeSampler:
            started = False

            def start(self):
                self.started = True

            def snapshot(self):
                return {"cpu": 25.0, "gpu": 0.0, "memory": 0.5}

        fake = FakeSampler()
        with mock.patch.object(system_stats, "_sampler", fake), \
                mock.patch.object(system_stats, "_read_ram_kb", return_value=(1000, 100)), \
                mock.patch.object(system_stats, "_read_temp_c", return_value=None), \
                mock.patch.object(system_stats, "_read_storage_bytes",
                                  return_value=(None, None)), \
                mock.patch.object(system_stats, "_read_npu_load_percent",
                                  return_value=None):
            raw = system_stats.collect_system_metrics()

        self.assertTrue(fake.started)
        self.assertEqual(raw["cpu"]["load"], 25.0)
        self.assertEqual(raw["gpu"]["load"], 0.0)
        # The averaged used fraction wins over total-available (which would be 0.9).
        self.assertEqual(raw["ram"]["usedFraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
