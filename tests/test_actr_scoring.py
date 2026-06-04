"""Unit tests for ACT-R base-level calibration and status classification.

Run with the project venv:  ./venv/bin/python -m unittest discover -s tests
These tests use only the standard library so they need no extra dependencies.
"""

import math
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from actr_scoring import (  # noqa: E402
    DEFAULT_TIME_UNIT_SECONDS,
    compute_base_level,
    compute_spreading_activation,
)
from forgetting import classify_memory_status  # noqa: E402

DAY = 86400.0


def _ago(days):
    return datetime.now(timezone.utc) - timedelta(days=days)


class BaseLevelCalibrationTest(unittest.TestCase):
    """Guards against the seconds-vs-days mis-calibration bug.

    With the day-scale unit, a single-access memory should stay retrievable for weeks,
    not seconds. The original code measured age in raw seconds, which pushed every memory
    past the -2 forgetting threshold within about a minute.
    """

    def test_recent_single_access_is_active(self):
        b = compute_base_level([_ago(0.04)], _ago(0.04))  # ~1 hour old
        self.assertGreater(b, 0.0)
        self.assertEqual(classify_memory_status(b), "active")

    def test_week_old_single_access_is_dormant_not_forgotten(self):
        b = compute_base_level([_ago(10)], _ago(10))
        self.assertTrue(-2.0 < b <= 0.0, f"B={b} should be dormant range")
        self.assertEqual(classify_memory_status(b), "dormant")

    def test_two_month_old_single_access_is_forgotten(self):
        b = compute_base_level([_ago(60)], _ago(60))
        self.assertLessEqual(b, -2.0)
        self.assertEqual(classify_memory_status(b), "forgotten")

    def test_more_accesses_raise_base_level(self):
        once = compute_base_level([_ago(3)], _ago(3))
        thrice = compute_base_level([_ago(3), _ago(3), _ago(3)], _ago(3))
        self.assertGreater(thrice, once)
        # three identical-age accesses add exactly ln(3) over one
        self.assertAlmostEqual(thrice - once, math.log(3), places=6)

    def test_seconds_unit_regresses_to_the_old_bug(self):
        """A raw-seconds unit forgets a minutes-old memory: the behavior we fixed."""
        ts = [_ago(1.0 / 24)]  # 1 hour old
        b_days = compute_base_level(ts, ts[0], time_unit_seconds=DAY)
        b_seconds = compute_base_level(ts, ts[0], time_unit_seconds=1.0)
        self.assertEqual(classify_memory_status(b_days), "active")
        self.assertEqual(classify_memory_status(b_seconds), "forgotten")
        # the two differ by the pure constant shift d*ln(unit), d=0.5
        self.assertAlmostEqual(b_days - b_seconds, 0.5 * math.log(DAY), places=6)

    def test_default_unit_is_one_day(self):
        self.assertEqual(DEFAULT_TIME_UNIT_SECONDS, DAY)


class SpreadingActivationTest(unittest.TestCase):
    def test_rarer_shared_tag_gives_stronger_signal(self):
        rare = compute_spreading_activation(["actr"], ["actr"], {"actr": 1}, S=2.0)
        common = compute_spreading_activation(["postgres"], ["postgres"], {"postgres": 50}, S=2.0)
        self.assertGreater(rare, common)

    def test_no_shared_tags_is_zero(self):
        self.assertEqual(
            compute_spreading_activation(["a"], ["b"], {"a": 1, "b": 1}, S=2.0), 0.0
        )


if __name__ == "__main__":
    unittest.main()
