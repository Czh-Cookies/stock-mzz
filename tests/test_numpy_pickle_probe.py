import copy
import io
import json
import os
from pathlib import Path
import pickle
import pickletools
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import zstandard as zstd

from src.numpy_pickle_probe import (convert_probe, NumericIndex, validate_index,
    cleanup_owned, MAX_SOURCE, ProtocolError)
from src.oof import load_protocol
from src.prefix_store import PrefixStore, KEYS

ROOT = Path(__file__).resolve().parents[1]


def fixture(t=14, s=5, n=3):
    rng = np.random.default_rng(42)
    return {'num_x':rng.standard_normal((t,s,n)).astype('<f4'),
        'cat_x':rng.integers(0,5,(t,s,2),dtype='<i8'), 'mask_x':rng.random((t,s))>.1,
        'y1':rng.standard_normal((t,s)).astype('<f4'), 'mask_y':rng.random((t,s))>.1,
        'y2':rng.standard_normal((t,s)).astype('<f4'),
        'train_start_idx':1, 'valid_start_idx':10, 'test_start_idx':12}


def wrapper(path, raw, checksum=True):
    blob=zstd.ZstdCompressor(level=1, write_checksum=checksum).compress(raw)
    path.write_bytes(b'\x80\x05B'+struct.pack('<I',len(blob))+blob+b'\x94.')


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.source=self.root/'synthetic.z'; self.dest=self.root/'prefix'
        self.data=fixture()
        self.p=load_protocol(ROOT/'configs/oof_r11.json')
        self.p.update(official_bounds=[1,10,12,14],purge_steps=1,
                      folds={'F1':[4,6], 'F2':[6,8], 'F3':[8,10]})
        self.raw=pickle.dumps(self.data,protocol=4)
        wrapper(self.source,self.raw)

    def tearDown(self): self.tmp.cleanup()

    def assert_clean(self):
        self.assertFalse(self.dest.exists())
        self.assertEqual(list(self.root.glob('.numpy-probe-*')),[])

    def test_roundtrip_values_dtype_shape_axes_and_store_compatibility(self):
        self.data['num_x'][0,0,:]=[np.nan,np.inf,-0.0]
        self.data['y1'][0,0]=-0.0
        wrapper(self.source,pickle.dumps(self.data,protocol=4))
        with patch('pickle.load',side_effect=AssertionError('forbidden')), \
             patch('pickle.loads',side_effect=AssertionError('forbidden')):
            info=convert_probe(self.source,self.dest,self.p)
        store=PrefixStore(self.dest,self.p,expected_source=info['source_sha256'])
        try:
            for key in KEYS:
                arr=store._arrays[key]; expected=self.data[key][:10]
                self.assertEqual(arr.dtype,expected.dtype)
                self.assertEqual(arr.shape,expected.shape)
                self.assertEqual(arr.tobytes(),expected.tobytes())
            self.assertEqual(store.contract['axes']['stock_ids'],list(map(str,range(5))))
            self.assertEqual(store.contract['axes']['num_feature_ids'],['0','1','2'])
            self.assertEqual(store.contract['axes']['cat_feature_ids'],['0','1'])
        finally: store.close()
        self.assertFalse((self.dest/'y2.npy').exists())

    def test_big_endian_preserves_exact_bytes(self):
        self.data['num_x']=self.data['num_x'].astype('>f8')
        wrapper(self.source,pickle.dumps(self.data,protocol=4))
        convert_probe(self.source,self.dest,self.p)
        out=np.load(self.dest/'num_x.npy',allow_pickle=False)
        self.assertEqual(out.dtype,self.data['num_x'].dtype)
        self.assertEqual(out.tobytes(),self.data['num_x'][:10].tobytes())

    def test_multi_chunk_rows_and_frame_without_checksum(self):
        data=fixture(s=500,n=600)
        wrapper(self.source,pickle.dumps(data,protocol=4),checksum=False)
        convert_probe(self.source,self.dest,self.p)
        out=np.load(self.dest/'num_x.npy',allow_pickle=False)
        self.assertEqual(out.tobytes(),data['num_x'][:10].tobytes())

    def test_corrupted_output_refused_before_publish(self):
        def damage(stage,work):
            if stage=='ready':
                with (work/'package/num_x.npy').open('r+b') as f:
                    f.seek(-1,2); byte=f.read(1); f.seek(-1,2); f.write(bytes([byte[0]^255]))
        with self.assertRaises(ProtocolError):convert_probe(self.source,self.dest,self.p,hook=damage)
        self.assert_clean()

    def test_metadata_parser_never_reads_array_payload(self):
        index=NumericIndex(io.BytesIO(self.raw)).parse()
        arrays=validate_index(index,self.p)
        spans=[(a.blob.offset,a.blob.offset+a.blob.length) for a in arrays.values()]
        class Guarded(io.BytesIO):
            def read(stream,n=-1):
                pos=stream.tell()
                if n<0 or n>128: raise AssertionError('unbounded read')
                if any(pos<b and pos+n>a for a,b in spans): raise AssertionError('array payload read')
                return super().read(n)
        validate_index(NumericIndex(Guarded(self.raw)).parse(),self.p)

    def test_binbytes8_and_unframed_payload(self):
        data=fixture(t=100,s=100,n=3)
        raw=pickle.dumps(data,protocol=4)
        # Only inspect this bounded synthetic pickle; widen its first unframed BINBYTES header.
        pos=next(pos for op,arg,pos in pickletools.genops(raw) if op.name=='BINBYTES')
        n=struct.unpack('<I',raw[pos+1:pos+5])[0]
        widened=raw[:pos]+b'\x8e'+struct.pack('<Q',n)+raw[pos+5:]
        original=NumericIndex(io.BytesIO(raw)).parse()['num_x']
        result=NumericIndex(io.BytesIO(widened)).parse()['num_x']
        self.assertEqual(result.shape,original.shape)
        self.assertEqual(result.blob.length,original.blob.length)
        self.assertEqual(result.blob.offset,original.blob.offset+4)

    def test_bad_frames_truncation_trailing_and_opcodes_rejected(self):
        frame_len=struct.unpack('<Q',self.raw[3:11])[0]
        variants=[self.raw[:-1],self.raw+b'extra',b'\x80\x04\x00',
                  self.raw[:3]+struct.pack('<Q',frame_len-3)+self.raw[11:],
                  b'\x80\x04\x8e'+struct.pack('<Q',2**40)]
        for raw in variants:
            with self.subTest(rawlen=len(raw)),self.assertRaises(ProtocolError):
                NumericIndex(io.BytesIO(raw)).parse()

    def test_unsupported_structures_refused_before_publication(self):
        class Unexpected:
            def __reduce__(self): return (str,('forbidden global',))
        variants=[]
        for value in (np.asfortranarray(self.data['num_x']), np.array([object()],dtype=object), Unexpected()):
            variants.append({**self.data,'num_x':value})
        variants.extend([{**self.data,'extra':'unknown'}, {**self.data,'valid_start_idx':9},
                         {**self.data,'mask_y':self.data['mask_x']}])
        for data in variants:
            wrapper(self.source,pickle.dumps(data,protocol=4))
            with self.assertRaises(ProtocolError): convert_probe(self.source,self.dest,self.p)
            self.assert_clean()

    def test_compressed_damage_outer_truncation_rejected(self):
        original=self.source.read_bytes()
        damaged=bytearray(original); damaged[-5]^=0xff
        for content in (original[:-1],original+b'x',bytes(damaged)):
            self.source.write_bytes(content)
            with self.assertRaises((ProtocolError,zstd.ZstdError)):
                convert_probe(self.source,self.dest,self.p)
            self.assert_clean()

    def test_interruption_cleanup_and_rerun(self):
        for stage in ('inflated','ready'):
            def stop(s,w):
                self.assertFalse(self.dest.exists())
                if s==stage: raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                convert_probe(self.source,self.dest,self.p,hook=stop)
            self.assert_clean()
        convert_probe(self.source,self.dest,self.p)
        with self.assertRaises(FileExistsError):convert_probe(self.source,self.dest,self.p)

    def test_atomic_publication_and_destination_race(self):
        def race(stage,work):
            self.assertFalse(self.dest.exists())
            if stage=='ready':
                self.assertTrue((work/'package/manifest.json').is_file())
                self.dest.mkdir(); (self.dest/'keep').write_text('existing')
        with self.assertRaises(FileExistsError):convert_probe(self.source,self.dest,self.p,hook=race)
        self.assertEqual((self.dest/'keep').read_text(),'existing')
        self.assertEqual(list(self.root.glob('.numpy-probe-*')),[])

    def test_hard_exit_orphan_cleanup_then_rerun(self):
        code='''import os,json,sys
from src.numpy_pickle_probe import convert_probe
def stop(stage,work):
    if stage=='ready': os._exit(19)
convert_probe(sys.argv[1],sys.argv[2],json.loads(sys.argv[3]),hook=stop)
'''
        result=subprocess.run([sys.executable,'-B','-c',code,str(self.source),str(self.dest),json.dumps(self.p)],cwd=ROOT)
        self.assertEqual(result.returncode,19)
        self.assertFalse(self.dest.exists())
        orphan,=self.root.glob('.numpy-probe-*')
        with self.assertRaises(ProtocolError):cleanup_owned(orphan,self.root/'different')
        cleanup_owned(orphan,self.dest)
        self.assert_clean()
        convert_probe(self.source,self.dest,self.p)

    def test_real_size_cap_and_oversized_declared_frame(self):
        with patch.object(Path,'stat') as stat:
            stat.return_value.st_size=MAX_SOURCE+1
            with self.assertRaises(ProtocolError):convert_probe(self.source,self.dest,self.p)
        raw=b'\x80\x04\x95'+struct.pack('<Q',2**40)
        with self.assertRaises(ProtocolError):NumericIndex(io.BytesIO(raw)).parse()
        bad_protocol=copy.deepcopy(self.p)
        bad_protocol['official_bounds']=[1,20,12,14]
        with self.assertRaises(ProtocolError):
            validate_index(NumericIndex(io.BytesIO(self.raw)).parse(),bad_protocol)
        self.assert_clean()


if __name__=='__main__':unittest.main()
