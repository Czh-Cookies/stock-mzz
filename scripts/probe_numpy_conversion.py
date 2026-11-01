"""Generate and benchmark small synthetic wrappers only. No --source option."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Conversion has no BLAS work; avoid needless native thread-pool commit in this tool.
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import numpy as np
import zstandard as zstd
from src.numpy_pickle_probe import convert_probe


def memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in (
            'peak_working_set','working_set','peak_paged_pool','paged_pool','peak_nonpaged_pool',
            'nonpaged_pool','commit','peak_commit','private')]
    c=Counters(); c.cb=ctypes.sizeof(c)
    process=ctypes.c_void_p(-1)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(process,ctypes.byref(c),c.cb):
        raise RuntimeError('cannot measure process memory')
    return {'peak_working_set':c.peak_working_set,'working_set':c.working_set,
            'private_bytes':c.private,'peak_commit_bytes':c.peak_commit,'page_faults':c.faults}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,help='new directory for synthetic files/report')
    p.add_argument('--rows',type=int,default=384,choices=[48,384])
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(); out=Path(a.out).resolve()
    if a.worker:
        # Worker accepts ONLY its parent's bounded synthetic fixture, not arbitrary paths.
        settings=json.loads((out/'fixture.json').read_text())
        before=memory(); start=time.perf_counter()
        result=convert_probe(out/'synthetic.z',out/'prefix',settings['protocol'])
        result.update(before=before,after=memory(),seconds=time.perf_counter()-start,
                      source_bytes=(out/'synthetic.z').stat().st_size,
                      native_threads=1)
        with (out/'measurement.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
        print(json.dumps(result)); return
    out.mkdir(parents=True,exist_ok=False)
    t,s,n=a.rows,256,99
    cutoff=int(t*.8)
    protocol={'official_bounds':[1,cutoff,t-1,t]}
    rng=np.random.default_rng(20261101)
    data={'num_x':rng.standard_normal((t,s,n),dtype=np.float32),
          'cat_x':rng.integers(0,100,(t,s,9),dtype=np.int64),
          'mask_x':rng.random((t,s))>.1,'mask_y':rng.random((t,s))>.1,
          'y1':rng.standard_normal((t,s),dtype=np.float32),
          'y2':rng.standard_normal((t,s),dtype=np.float32),
          'train_start_idx':1,'valid_start_idx':cutoff,'test_start_idx':t-1}
    raw=pickle.dumps(data,protocol=4)
    blob=zstd.ZstdCompressor(level=1,write_checksum=True).compress(raw)
    (out/'synthetic.z').write_bytes(b'\x80\x05B'+struct.pack('<I',len(blob))+blob+b'\x94.')
    (out/'fixture.json').write_text(json.dumps({'protocol':protocol,'seed':20261101,
        'generator':'numpy numeric protocol4; exact observed outer wrapper','rows':t}),encoding='utf-8')
    # Fixture creation is deliberately outside the measured conversion process.
    del raw,blob,data
    subprocess.run([sys.executable,'-B',__file__,'--out',str(out),'--worker'],check=True)


if __name__=='__main__':main()
