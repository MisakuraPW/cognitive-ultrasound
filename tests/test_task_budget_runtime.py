"""Dependency-light regression tests for lossless ephemeral exchange."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from cognitive_ultrasound.task_budget.exchange import write_exchange


class ExchangeTests(unittest.TestCase):
    def test_modes_preserve_changed_arrays_and_do_not_alias(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d)/"exchange.npz"
            for mode in ("compressed", "uncompressed", "tmpfs"):
                old = None
                for value in (1, 2):
                    a = np.arange(96, dtype=np.float32).reshape(2,3,4,4)*value
                    write_exchange(file, dict(clips=a, predictions=np.array([value],np.float32)), mode)
                    with np.load(file, allow_pickle=False) as f:
                        loaded = f["clips"].copy()
                    self.assertEqual(loaded.dtype, a.dtype)
                    self.assertEqual(loaded.tobytes(), a.tobytes())
                    if old is not None:
                        self.assertEqual(old.tobytes(), (a/2).tobytes())
                    old = loaded
                self.assertFalse(file.with_suffix(".tmp").exists())

    def test_unknown_mode_is_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d)/"exchange.npz"
            with self.assertRaises(ValueError):
                write_exchange(file, dict(x=np.ones(2)), "silent_fallback")
            self.assertFalse(file.exists())


if __name__ == "__main__":
    unittest.main()
