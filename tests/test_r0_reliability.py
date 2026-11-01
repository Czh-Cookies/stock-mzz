"""R0 acceptance tests: synthetic panels only; unittest needs no extra packages."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util

import numpy as np
import torch

from src.dataset import dump_split_cache, load_split_cache, load_eval_splits
from src.provenance import CacheValidationError, panel_contract
from src.metrics import mean_rank_ic
from src.models.gru_ts import GRUModel
from src.models.fusion import (blend_in_space, linear_blend, panel_rank_mean,
                               n2000_hard_gate_mlp, coverage_stock_three_way)


def panel():
    rng = np.random.default_rng(8)
    return dict(num_x=rng.normal(size=(9, 12, 4)).astype(np.float32),
                cat_x=rng.integers(0, 3, (9, 12, 2)),
                mask_x=np.ones((9, 12), bool), mask_y=np.ones((9, 12), bool),
                y1=rng.random((9, 12)).astype(np.float32),
                train_start_idx=2, valid_start_idx=5, test_start_idx=7,
                num_feature_ids=['a', 'b', 'c', 'd'], cat_feature_ids=['i', 'j'],
                stock_ids=list(range(12)), _source_sha256='a' * 64)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data = panel()
        self.cfg = dict(num_indices=[2, 0], length=3, include_current_day=False,
                        hidden_size=5, num_layers=1, cat_indices=[1], cat_embed_dim=2,
                        append_mask_channel=True, append_breadth_channel=True,
                        cs_hidden_norm=True, clip=2., industry_col=0, loss_weight='coverage',
                        target_space='rank', rank_loss_weight=.2)

    def model(self):
        m = GRUModel(self.cfg)
        m.device = torch.device('cpu')
        m.prepare_features(self.data)
        m.net = m._build_net()
        return m

    def predict(self, m):
        return m.predict_panel({'start': 5, 'mask_x': self.data['mask_x'][5:7]}, self.data)

    def test_roundtrip_rebuilds_all_channels_and_discards_stale_features(self):
        m = self.model()
        expected = self.predict(m)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'model.pt'
            m.save(p)
            other = GRUModel(dict(num_indices=[1], hidden_size=2))
            other.device = torch.device('cpu')
            other.prepare_features(self.data)
            other.load(p)
            self.assertIsNone(other.cs_sel)
            self.assertEqual(other.feature_spec(), m.feature_spec())
            self.assertEqual(other.loss_weight, m.loss_weight)
            self.assertEqual(other.rank_loss_weight, m.rank_loss_weight)
            np.testing.assert_array_equal(self.predict(other), expected)
            np.testing.assert_array_equal(self.predict(other), expected)

    def test_feature_order_change_rejected(self):
        m = self.model()
        m.cols.reverse()
        with self.assertRaisesRegex(ValueError, 'feature order'):
            self.predict(m)

    def test_named_input_feature_order_rejected(self):
        m = self.model()
        changed = dict(self.data)
        changed['num_feature_ids'] = ['b', 'a', 'c', 'd']
        with self.assertRaisesRegex(ValueError, 'axis order'):
            m.prepare_features(changed)

    def test_tiny_synthetic_fit_smoke(self):
        m = GRUModel({**self.cfg, 'max_epochs': 1})
        m.device = torch.device('cpu')
        result = m.fit(self.data, 2, 5)
        self.assertEqual(len(result['history']), 1)
        self.assertTrue(np.isfinite(self.predict(m)).all())

    def test_weights_metadata_mismatch_rejected(self):
        m = self.model()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'model.pt'
            m.save(p)
            with p.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                m.load(p)

    def test_legacy_cfg_restores_added_channels(self):
        m = self.model()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'model.pt'
            m.save(p)
            mp = p.with_suffix('.pt.meta.json')
            meta = json.loads(mp.read_text())
            meta.pop('feature_spec'); meta.pop('format_version')
            mp.write_text(json.dumps(meta))
            other = GRUModel(dict(num_indices=[0]))
            other.device = torch.device('cpu')
            other.load(p)
            np.testing.assert_array_equal(self.predict(other), self.predict(m))

    def test_missing_metadata_and_feature_override_rejected(self):
        m = self.model()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'model.pt'
            with self.assertRaisesRegex(ValueError, 'metadata missing'):
                m.load(p)
            m.save(p)
            with self.assertRaisesRegex(ValueError, 'not a training override'):
                m.load(p, training_overrides={'num_indices': [0]})
            m.load(p, training_overrides={'freeze_backbone': True, 'max_epochs': 1})
            self.assertTrue(m.freeze_backbone)
            self.assertEqual(m.cfg['max_epochs'], 1)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.data = panel()
        self.contract = panel_contract(self.data, 0)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        dump_split_cache(self.data, self.root, industry_col=0)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, contract=None):
        return load_split_cache('valid', self.root, industry_col=0,
                                expected_contract=self.contract if contract is None else contract)

    def test_valid_roundtrip_and_no_test_labels(self):
        actual = self.read()
        np.testing.assert_array_equal(actual['y1'], self.data['y1'][5:7])
        with np.load(self.root / 'test.npz') as z:
            self.assertNotIn('y1', z.files)
            self.assertNotIn('mask_y', z.files)
        out, _ = load_eval_splits(cache_dir=self.root, expected_contract=self.contract,
                                  industry_col=0, splits=('valid',))
        np.testing.assert_array_equal(out['valid']['y1'], actual['y1'])

    def test_source_split_features_stock_order_and_preprocessing_rejected(self):
        variants = []
        for key, value in [('source_sha256', 'b'*64), ('bounds', [2, 4, 7, 9]),
                           ('num_features', ['b', 'a', 'c', 'd']),
                           ('cat_features', ['j', 'i']), ('stock_order', list(reversed(self.contract['stock_order'])))]:
            c = copy.deepcopy(self.contract); c[key] = value; variants.append(c)
        c = copy.deepcopy(self.contract); c['preprocessing']['version'] = 2; variants.append(c)
        for c in variants:
            with self.subTest(contract=c), self.assertRaises(CacheValidationError):
                self.read(c)

    def test_payload_tampering_rejected(self):
        p = self.root / 'valid.npz'
        with np.load(p) as z:
            payload = {k: z[k] for k in z.files}
        payload['mask_x'][0, 0] = False
        np.savez(p, **payload)
        with self.assertRaisesRegex(CacheValidationError, 'payload mismatch'):
            self.read()

    def test_legacy_cache_rejected(self):
        np.savez(self.root / 'valid.npz', y1=self.data['y1'][5:7])
        with self.assertRaisesRegex(CacheValidationError, 'legacy cache'):
            self.read()

    def test_missing_cache_no_implicit_rebuild(self):
        with self.assertRaises(CacheValidationError):
            load_eval_splits(cache_dir=self.root/'absent', expected_contract=self.contract,
                             dump_if_missing=False, splits=('valid',))

    def test_source_hash_mismatch_rejected(self):
        with self.assertRaisesRegex(CacheValidationError, 'source mismatch'):
            load_split_cache('valid', self.root, source_sha256='b'*64, industry_col=0)

    def test_existing_cache_not_overwritten(self):
        before = (self.root/'valid.npz').read_bytes()
        with self.assertRaisesRegex(CacheValidationError, 'overwrite'):
            dump_split_cache(self.data, self.root, industry_col=0)
        self.assertEqual(before, (self.root/'valid.npz').read_bytes())


class ScaleAndMetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.a = rng.normal(size=(5, 20))
        self.b = rng.normal(size=(5, 20))
        self.mask = np.ones((5, 20), bool)

    def test_rank_and_zscore_are_invariant_to_extreme_positive_scale(self):
        for space in ('rank', 'zscore'):
            with self.subTest(space=space):
                a = blend_in_space(self.a, self.b, self.mask, .3, space)
                b = blend_in_space(self.a*1e12+1e13, self.b*1e-12, self.mask, .3, space)
                np.testing.assert_allclose(a, b, atol=2e-6)

    def test_mixed_rank_raw_rejected_and_explicit_conversion_works(self):
        r = panel_rank_mean([self.a, self.b], self.mask)
        with self.assertRaisesRegex(ValueError, 'mixed score spaces'):
            linear_blend(r, self.b, .25)
        with self.assertRaisesRegex(ValueError, 'mixed score spaces'):
            coverage_stock_three_way(r, self.b, None, self.mask, tau=10, temperature=1,
                                       weight_low=.25, weight_high=.4)
        out = n2000_hard_gate_mlp(r, self.b, self.a, self.mask, 10, space='rank')
        self.assertTrue(np.isfinite(out).all())

    def test_metric_rejects_nonfinite_in_valid_positions(self):
        for bad in (np.nan, np.inf, -np.inf):
            for target in ('prediction', 'label'):
                with self.subTest(bad=bad, target=target):
                    p, y = self.a.copy(), self.b.copy()
                    (p if target == 'prediction' else y)[0, 0] = bad
                    with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
                        mean_rank_ic(p, y, self.mask)
                    mask = self.mask.copy(); mask[0, 0] = False
                    self.assertTrue(np.isfinite(mean_rank_ic(p, y, mask)))

    def test_metric_mask_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mask shape'):
            mean_rank_ic(self.a, self.b, self.mask[:1])


class StatsCacheTests(unittest.TestCase):
    def test_collapse_stats_rejects_legacy_and_changed_contract(self):
        spec = importlib.util.spec_from_file_location('r0_collapse',
            Path(__file__).resolve().parents[1] / 'scripts/eval_train_collapse_switch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'stats.npz'
            with patch.object(module, 'STATS_TRAIN', path), patch.object(
                    module, '_stats_contract', return_value={'source': 'expected'}):
                np.savez(path, n_x=np.arange(3))
                with self.assertRaisesRegex(CacheValidationError, 'unverified'):
                    module._fit_lock()
                np.savez(path, manifest=np.array(json.dumps({'source': 'other'})))
                with self.assertRaisesRegex(CacheValidationError, 'mismatch'):
                    module._fit_lock()


if __name__ == '__main__':
    unittest.main()
