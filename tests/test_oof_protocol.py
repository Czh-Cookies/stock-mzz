import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from src.oof import (ProtocolError, load_protocol, fold_spec, label_lag, prefix_contract,
    run_contract, new_run, prepare_fold, write_label_cache, load_label_cache,
    register_checkpoint, verify_checkpoint, save_predictions, evaluate_run, eligible_fusion_rows)

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p = load_protocol(ROOT/'configs/oof_r11.json')
        self.p['official_bounds'] = [2, 20, 24, 28]
        self.p['folds'] = {'F1': [8, 12], 'F2': [12, 16], 'F3': [16, 20]}
        self.p['purge_steps'] = 2
        self.p['label'].update(horizon_steps=2, availability_lag_steps=0)
        rng = np.random.default_rng(4)
        self.data = {'num_x': rng.normal(size=(20, 6, 3)).astype('float32'),
                     'cat_x': np.zeros((20, 6, 1), 'int64'),
                     'y1': rng.random((20, 6)).astype('float32'),
                     'mask_x': np.ones((20, 6), bool), 'mask_y': np.ones((20, 6), bool)}
        self.contract = prefix_contract(self.data, 'a'*64, self.p)
        self.run = self.make_run()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_run(self, fold='F1', **kw):
        return run_contract(self.p, fold, self.contract, 'toy', {'features': [2, 0]},
                            {'toy.py': 'b'*64}, **kw)

    def saved(self):
        dest = new_run(self.root, self.run)
        cp = dest/'model.bin'; cp.write_bytes(b'synthetic model, no training')
        register_checkpoint(dest, cp, self.run, fitted_range=self.run['train'])
        return dest, cp

    def test_actual_fold_boundaries(self):
        p = load_protocol(ROOT/'configs/oof_r11.json')
        for fold, end, recent in [('F1', 2169, 1369), ('F2', 2412, 1612), ('F3', 2655, 1855)]:
            self.assertEqual(fold_spec(p, fold)['train'], [486, end])
            self.assertEqual(fold_spec(p, fold, 800)['train'], [recent, end])

    def test_unknown_horizon_blocks_certification(self):
        p = load_protocol(ROOT/'configs/oof_r11.json')
        with self.assertRaisesRegex(ProtocolError, 'unknown'):
            label_lag(p)
        self.assertEqual(label_lag(p, exploratory=True), 20)

    def test_known_lag_must_fit_gap(self):
        self.p['label']['availability_lag_steps'] = 1
        with self.assertRaisesRegex(ProtocolError, 'shorter'):
            label_lag(self.p)

    def test_full_panel_and_y2_rejected(self):
        with self.assertRaises(ProtocolError):
            prefix_contract({**self.data, 'y2': self.data['y1']}, 'a'*64, self.p)
        with self.assertRaises(ProtocolError):
            prefix_contract({**self.data, 'y1': np.zeros((28, 6))}, 'a'*64, self.p)

    def test_trainer_labels_stop_before_purge_and_eval_has_no_labels(self):
        train, evaluate, labels = prepare_fold(self.data, self.run)
        self.assertEqual(train['y1'].shape[0], 6)
        self.assertEqual(train['end'], 6)
        self.assertFalse(train['mask_y'][:2].any())
        self.assertTrue(np.isnan(train['y1'][:2]).all())
        self.assertNotIn('y1', evaluate)
        self.assertNotIn('mask_y', evaluate)
        np.testing.assert_array_equal(labels['y1'], self.data['y1'][8:12])
        self.assertFalse(np.shares_memory(train['y1'], self.data['y1']))

    def test_changed_source_axes_payload_rejected(self):
        d = dict(self.data); d['stock_ids'] = list(reversed(range(6)))
        with self.assertRaisesRegex(ProtocolError, 'mismatch'):
            prepare_fold(d, self.run)
        d = copy.deepcopy(self.data); d['num_x'][0, 0, 0] += 1
        with self.assertRaises(ProtocolError):
            prepare_fold(d, self.run)

    def test_old_checkpoint_initialization_rejected(self):
        with self.assertRaises(ProtocolError):
            run_contract(self.p, 'F1', self.contract, 'toy', {'paths': {'init_checkpoint': 'old.pt'}}, {'a': 'b'*64})

    def test_overlap_and_selection_rejected(self):
        bad = copy.deepcopy(self.run); bad['train'][1] = 9
        with self.assertRaises(ProtocolError):
            prepare_fold(self.data, bad)
        dest = new_run(self.root, self.run); cp = dest/'model.bin'; cp.write_bytes(b'x')
        with self.assertRaises(ProtocolError):
            register_checkpoint(dest, cp, self.run, fitted_range=[2, 20])
        with self.assertRaises(ProtocolError):
            register_checkpoint(dest, cp, self.run, fitted_range=[2, 6], selection_range=[8, 12])

    def test_run_directory_and_foreign_checkpoint_guard(self):
        dest = new_run(self.root, self.run)
        with self.assertRaises(FileExistsError):
            new_run(self.root, self.run)
        cp = self.root/'old.bin'; cp.write_bytes(b'old')
        with self.assertRaises(ProtocolError):
            register_checkpoint(dest, cp, self.run, fitted_range=[2, 6])

    def test_fold_cache_mismatch_and_no_overwrite(self):
        dest, _ = self.saved()
        _, _, labels = prepare_fold(self.data, self.run)
        path = write_label_cache(dest, labels, self.run)
        with self.assertRaises(FileExistsError):
            write_label_cache(dest, labels, self.run)
        with self.assertRaises(ProtocolError):
            load_label_cache(path, self.make_run('F2'))

    def test_checkpoint_mutation_rejected(self):
        dest, cp = self.saved(); cp.write_bytes(b'changed')
        with self.assertRaises(ProtocolError):
            verify_checkpoint(dest, self.run)

    def test_complete_synthetic_artifact_chain(self):
        dest, _ = self.saved()
        _, _, labels = prepare_fold(self.data, self.run)
        write_label_cache(dest, labels, self.run)
        path = save_predictions(dest, labels['y1'].copy(), self.run, score_space='raw')
        self.assertAlmostEqual(evaluate_run(dest)['score'], 1.0)
        with self.assertRaises(FileExistsError):
            save_predictions(dest, labels['y1'], self.run, score_space='raw')
        with path.open('ab') as f: f.write(b'tamper')
        with self.assertRaises(ProtocolError):
            evaluate_run(dest)

    def test_fusion_maturity_cutoff(self):
        # At time 12, row 10 matures at 12 and is not yet available (< cutoff).
        np.testing.assert_array_equal(eligible_fusion_rows(self.p, 'F2', self.run), [8, 9])
        with self.assertRaises(ProtocolError):
            eligible_fusion_rows(self.p, 'F1', self.run)
        with self.assertRaises(ProtocolError):
            eligible_fusion_rows(self.p, 'F2', self.make_run('F3'))

    def test_nonfinite_prediction_rejected(self):
        dest, _ = self.saved()
        with self.assertRaises(ProtocolError):
            save_predictions(dest, np.full((4, 6), np.nan), self.run, score_space='raw')


if __name__ == '__main__':
    unittest.main()
