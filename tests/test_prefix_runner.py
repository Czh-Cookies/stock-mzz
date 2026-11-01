import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.oof import load_protocol, ProtocolError, evaluate_run
from src.prefix_store import PrefixStore, export_prefix, KEYS
from src.fold_runner import train_fold, fixed_fusion, fuse_runs, check_memory_budget

ROOT = Path(__file__).resolve().parents[1]


class PrefixRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.p = load_protocol(ROOT/'configs/oof_r11.json')
        self.p.update(official_bounds=[2,20,24,28], purge_steps=2,
                      folds={'F1':[8,12],'F2':[12,16],'F3':[16,20]})
        self.p['label'].update(horizon_steps=2, availability_lag_steps=0)
        r = np.random.default_rng(4)
        self.data = {'num_x':r.normal(size=(20,12,4)).astype('float32'),
                     'cat_x':r.integers(0,3,(20,12,2),dtype='int64'),
                     'y1':r.random((20,12)).astype('float32'),
                     'mask_y':np.ones((20,12),bool),'mask_x':np.ones((20,12),bool),
                     'stock_ids':list(range(12)), 'num_feature_ids':['a','b','c','d'],
                     'cat_feature_ids':['i','j']}
        export_prefix(self.data,'a'*64,self.p,self.root/'store',chunk_rows=3)
        self.store = PrefixStore(self.root/'store',self.p,expected_source='a'*64)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_roundtrip_dtype_shape_axes_values(self):
        for k in KEYS:
            a = self.store._arrays[k]
            self.assertIsInstance(a,np.memmap)
            self.assertEqual(a.dtype,self.data[k].dtype)
            np.testing.assert_array_equal(a,self.data[k])
        self.assertEqual(self.store.contract['axes']['num_feature_ids'],['a','b','c','d'])
        self.assertEqual(self.store.contract['axes']['stock_ids'],list(map(str,range(12))))

    def test_fold_label_permissions(self):
        for f in self.p['folds']:
            train=self.store.training_panel(f)
            self.assertFalse(train['mask_y'][:2].any())
            end=self.p['folds'][f][0]-2
            self.assertEqual(len(train['y1']),end)
            np.testing.assert_array_equal(train['y1'][2:],self.data['y1'][2:end])
            self.assertIsNone(train['y1'].base)
            self.assertNotIn('y1',self.store.evaluation_inputs(f)[0])
            with self.assertRaises(ProtocolError):
                self.store.label_window(f,'train',2,self.p['folds'][f][0])
            with self.assertRaises(ProtocolError):
                self.store.label_window(f,'evaluate',0,1)
        with self.assertRaises(ProtocolError): self.store.features(24)

    def test_axis_reordering_is_rejected(self):
        path=self.root/'store/manifest.json'
        manifest=json.loads(path.read_text())
        manifest['contract']['axes']['num_feature_ids'].reverse()
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ProtocolError):
            PrefixStore(self.root/'store',self.p,expected_source='a'*64)

    def test_memory_floor_and_protocol_mismatch(self):
        with self.assertRaises(ProtocolError):check_memory_budget('gru',2*2**30)
        self.assertEqual(check_memory_budget('gru',7*2**30),6*2**30)
        with self.assertRaises(ProtocolError):check_memory_budget('hist',9*2**30)
        changed=copy.deepcopy(self.p); changed['purge_steps']=3
        with self.assertRaises(ProtocolError):
            train_fold(self.store,changed,'F1','gru',self.root/'bad_protocol',execute=True)

    def test_source_mismatch_and_corrupt_array_rejected(self):
        with self.assertRaises(ProtocolError):
            PrefixStore(self.root/'store',self.p,expected_source='b'*64)
        with (self.root/'store/num_x.npy').open('ab') as f:f.write(b'bad')
        with self.assertRaises(ProtocolError):
            PrefixStore(self.root/'store',self.p,expected_source='a'*64)

    def test_export_refuses_existing_and_full_labels(self):
        with self.assertRaises(FileExistsError): export_prefix(self.data,'a'*64,self.p,self.root/'store')
        bad={**self.data,'y1':np.zeros((28,12),dtype='float32')}
        with self.assertRaises(ProtocolError):export_prefix(bad,'a'*64,self.p,self.root/'bad')

    def test_dry_run_never_opens_store_or_trains(self):
        result=train_fold(None,self.p,'F1','gru',self.root/'unused')
        self.assertEqual(result['default_action'],'dry-run')
        self.assertFalse((self.root/'unused').exists())

    def test_unknown_horizon_requires_explicit_acceptance(self):
        self.p['label']['horizon_steps']=None
        with self.assertRaises(ProtocolError):
            train_fold(self.store,self.p,'F1','gru',self.root/'blocked',execute=True)

    def test_four_adapters_synthetic_fit_and_artifact_chain(self):
        results={}; run_dirs={}
        for name in ('gru','mlp','baseline','hist'):
            cfg={'seed':42,'features':{'num_indices':[0,1], 'length':2,
                  'include_current_day':False,'cat_indices':[1], 'industry_col':0,
                  'max_train_stocks_per_day':12, 'include_raw':True, 'cs_zscore':False,
                  'industry_zscore':False},
                 'model':{'hidden_size':4,'max_epochs':1,'log_every_days':0,
                          'params':{'n_estimators':2,'num_leaves':3,'min_child_samples':2,'verbosity':-1,'n_jobs':1}},
                 'train':{'log_every_days':0}}
            if name=='hist':
                cfg['features']['history']={'enabled':True,'length':2,'source':'cs_zscore',
                                           'num_indices':[0], 'stats':['mean']}
            accessed=[]; original=self.store.label_window
            def guarded(fold,role,start,end,recent_days=None):
                accessed.append(role)
                return original(fold,role,start,end,recent_days)
            with patch.object(self.store,'label_window',side_effect=guarded):
                output=train_fold(self.store,self.p,'F1',name,self.root/name,execute=True,config=cfg)
            self.assertEqual(accessed,['train','evaluate'])
            dest=Path(output['run_dir'])
            run_dirs[name]=dest
            self.assertTrue(np.isfinite(evaluate_run(dest)['score']))
            results[name]=np.load(dest/'evaluation_predictions.npy')
            with self.assertRaises(FileExistsError):
                train_fold(self.store,self.p,'F1',name,self.root/name,execute=True,config=cfg)
        fused=fixed_fusion(results,np.ones((4,12),bool))
        self.assertEqual(fused.shape,(4,12))
        with patch.object(self.store,'label_window',side_effect=AssertionError('fusion cannot read labels')):
            path=fuse_runs(self.store,'F1',run_dirs,self.root/'fusion')
        np.testing.assert_array_equal(np.load(path/'evaluation_predictions.npy'),fused)
        with self.assertRaises(FileExistsError):fuse_runs(self.store,'F1',run_dirs,path)
        with self.assertRaises(ProtocolError):fuse_runs(self.store,'F2',run_dirs,self.root/'wrong_fold')
        with (run_dirs['gru']/'model.pt').open('ab') as f:f.write(b'corrupt')
        with self.assertRaises(ProtocolError):fuse_runs(self.store,'F1',run_dirs,self.root/'bad_fusion')

    def test_fixed_fusion_extreme_scale_invariance(self):
        r=np.random.default_rng(8)
        preds={k:r.normal(size=(3,12)) for k in ('gru','mlp','baseline','hist')}
        mask=np.ones((3,12),bool)
        scaled={k:v*(10.**(i*6)) for i,(k,v) in enumerate(preds.items())}
        np.testing.assert_array_equal(fixed_fusion(preds,mask),fixed_fusion(scaled,mask))


if __name__=='__main__':unittest.main()
