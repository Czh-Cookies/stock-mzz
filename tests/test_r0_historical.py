"""Read-only historical replay, NOT authorization to reuse unverified caches for research.

Only valid labels are read. Missing private artifacts skip these local integration
tests; synthetic R0 tests run on any checkout. No test-label path is opened.
"""
from pathlib import Path
import unittest

import numpy as np

from src.metrics import mean_rank_ic
from src.models.fusion import n2000_hard_gate_mlp


ROOT = Path(__file__).resolve().parents[1]


class HistoricalReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = [ROOT / 'outputs/split_cache/valid.npz'] + [
            ROOT / f'outputs/{name}_valid.npy' for name in
            ('gru_no_today_recent_n2000', 'fusion', 'cs_mlp', 'fusion_gru_ensemble')]
        if not all(p.is_file() for p in paths):
            raise unittest.SkipTest('private valid-only historical artifacts unavailable')
        # Deliberate audit-only access, kept out of production cache loaders.
        with np.load(paths[0], allow_pickle=False) as z:
            cls.y, cls.my, cls.mx = z['y1'], z['mask_y'], z['mask_x']
        cls.g, cls.t, cls.m, cls.saved = [np.load(p, allow_pickle=False) for p in paths[1:]]

    def test_0117177_legacy_formula(self):
        tau = float(np.quantile(self.mx.sum(1), .6))
        pred = n2000_hard_gate_mlp(self.g, self.t, self.m, self.mx, tau)
        self.assertAlmostEqual(mean_rank_ic(pred, self.y, self.my), 0.11717674392133287, places=12)

    def test_0122226_legacy_formula_and_exact_saved_predictions(self):
        pred = n2000_hard_gate_mlp(self.g, self.t, self.m, self.mx, 4588.4, .15, .7, .15)
        self.assertAlmostEqual(mean_rank_ic(pred, self.y, self.my), 0.12222639755275264, places=12)
        np.testing.assert_array_equal(pred, self.saved)


if __name__ == '__main__':
    unittest.main()
