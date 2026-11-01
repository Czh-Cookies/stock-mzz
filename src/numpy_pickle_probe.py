"""Synthetic-only feasibility probe, NOT a general unpickler or production converter.

No pickle.load/loads, imports from pickle symbols, callable REDUCE, or object BUILD.
Only a flat contest dict of C-order numeric NumPy protocol-4 arrays is recognized.
Real data.z is deliberately excluded by compressed/raw size caps in this phase.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np
import zstandard as zstd

from src.oof import ProtocolError, digest
from src.prefix_store import KEYS
from src.provenance import canonical, file_sha256

CHUNK = 1024 * 1024
MAX_SOURCE = 64 * CHUNK
MAX_RAW = 128 * CHUNK
MARK = object()
GLOBALS = {('numpy', 'ndarray'), ('numpy', 'dtype'),
           ('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', '_reconstruct')}


@dataclass(frozen=True)
class Symbol:
    module: str
    name: str


@dataclass(frozen=True)
class Blob:
    offset: int
    length: int


@dataclass
class Dtype:
    code: str
    value: object = None


@dataclass
class Array:
    shape: tuple = ()
    dtype: object = None
    blob: object = None


class NumericIndex:
    """Bounded symbolic grammar; binary payloads are seeked over, never unpickled."""
    def __init__(self, stream):
        self.f = stream
        self.size = stream.seek(0, 2)
        stream.seek(0)
        if self.size > MAX_RAW:
            raise ProtocolError('synthetic probe raw-size cap exceeded')
        self.frame_end = None
        self.stack, self.memo = [], []
        self.metadata_bytes = 0

    def take(self, n):
        if n > 128 or n < 0:
            raise ProtocolError('metadata field too large')
        self.bound(n)
        b = self.f.read(n)
        if len(b) != n:
            raise ProtocolError('truncated metadata')
        self.metadata_bytes += n
        if self.metadata_bytes > 65536:
            raise ProtocolError('metadata budget exceeded')
        return b

    def bound(self, n):
        end = self.f.tell() + n
        if end > self.size or (self.frame_end is not None and end > self.frame_end):
            raise ProtocolError('payload/frame boundary exceeds file or frame')

    def number(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def marked(self):
        pos = next((i for i in range(len(self.stack)-1, -1, -1) if self.stack[i] is MARK), None)
        if pos is None:
            raise ProtocolError('missing MARK')
        items = self.stack[pos+1:]
        del self.stack[pos:]
        return items

    def reduce(self, symbol, args):
        if not isinstance(symbol, Symbol) or type(args) is not tuple:
            raise ProtocolError('unsupported REDUCE')
        if symbol == Symbol('numpy', 'dtype'):
            codes = {f'{kind}{n}' for kind in 'iu' for n in (1, 2, 4, 8)} | {'f4', 'f8', 'b1'}
            if len(args) != 3 or args[0] not in codes or args[1:] != (False, True):
                raise ProtocolError('unsupported dtype constructor')
            return Dtype(args[0])
        if symbol.name == '_reconstruct':
            if (len(args) != 3 or args[:2] != (Symbol('numpy', 'ndarray'), (0,))
                    or not isinstance(args[2], Blob) or args[2].length != 1):
                raise ProtocolError('unsupported ndarray constructor')
            pos = self.f.tell()
            self.f.seek(args[2].offset)
            token = self.f.read(1)
            self.f.seek(pos)
            if token != b'b':
                raise ProtocolError('unsupported ndarray typecode')
            return Array()
        raise ProtocolError('callable not allowed')

    def build(self, obj, state):
        if type(state) is not tuple:
            raise ProtocolError('unsupported BUILD state')
        if isinstance(obj, Dtype):
            if (obj.value is not None or len(state) != 8 or state[0] != 3
                    or state[1] not in ('<', '>', '|', '=')
                    or state[2:] != (None, None, None, -1, -1, 0)):
                raise ProtocolError('unsupported dtype state')
            obj.value = np.dtype(state[1] + obj.code)
        elif isinstance(obj, Array):
            if (obj.blob is not None or len(state) != 5 or state[0] != 1
                    or type(state[1]) is not tuple or not 1 <= len(state[1]) <= 3
                    or any(type(x) is not int or not 0 < x <= 100000 for x in state[1])
                    or not isinstance(state[2], Dtype) or state[2].value is None
                    or state[3] is not False or not isinstance(state[4], Blob)):
                raise ProtocolError('unsupported array state/order/shape')
            shape, dtype, blob = state[1], state[2].value, state[4]
            if math.prod(shape) * dtype.itemsize != blob.length:
                raise ProtocolError('array shape/dtype/payload byte count mismatch')
            obj.shape, obj.dtype, obj.blob = shape, dtype, blob
        else:
            raise ProtocolError('object BUILD forbidden')

    def parse(self):
        try:
            if self.take(2) != b'\x80\x04':
                raise ProtocolError('only inner protocol 4 supported')
            for _ in range(4096):
                if self.frame_end == self.f.tell():
                    self.frame_end = None
                op = self.take(1)
                s = self.stack
                if op == b'\x95':
                    if self.frame_end is not None:
                        raise ProtocolError('nested/unfinished FRAME')
                    n = self.number('<Q')
                    if n > CHUNK or self.f.tell()+n > self.size:
                        raise ProtocolError('FRAME exceeds limit/file')
                    self.frame_end = self.f.tell()+n
                elif op == b'}': s.append({})
                elif op == b'(': s.append(MARK)
                elif op == b'N': s.append(None)
                elif op in (b'\x88', b'\x89'): s.append(op == b'\x88')
                elif op in (b'K', b'M', b'J'):
                    s.append(self.number({b'K':'<B', b'M':'<H', b'J':'<i'}[op]))
                elif op in (b'\x8c', b'X'):
                    n = self.number('<B' if op == b'\x8c' else '<I')
                    s.append(self.take(n).decode('utf-8'))
                elif op in (b'C', b'B', b'\x8e'):
                    n = self.number({b'C':'<B', b'B':'<I', b'\x8e':'<Q'}[op])
                    self.bound(n)
                    s.append(Blob(self.f.tell(), n))
                    self.f.seek(n, 1)
                elif op == b'\x94': self.memo.append(s[-1])
                elif op in (b'h', b'j'):
                    s.append(self.memo[self.number('<B' if op == b'h' else '<I')])
                elif op in (b'\x85', b'\x86', b'\x87'):
                    n = op[0]-0x84
                    if len(s) < n: raise ProtocolError('tuple stack underflow')
                    vals = tuple(s[-n:]); del s[-n:]; s.append(vals)
                elif op == b't': s.append(tuple(self.marked()))
                elif op == b'\x93':
                    name, module = s.pop(), s.pop()
                    if (module, name) not in GLOBALS:
                        raise ProtocolError('global symbol forbidden')
                    s.append(Symbol(module, name))
                elif op == b'R':
                    args, symbol = s.pop(), s.pop()
                    s.append(self.reduce(symbol, args))
                elif op == b'b':
                    state = s.pop(); self.build(s[-1], state)
                elif op == b'u':
                    items = self.marked()
                    if not s or type(s[-1]) is not dict or len(items) % 2:
                        raise ProtocolError('invalid SETITEMS')
                    for key, value in zip(items[::2], items[1::2]):
                        if type(key) is not str or key in s[-1]:
                            raise ProtocolError('invalid/duplicate dictionary key')
                        s[-1][key] = value
                elif op == b'.':
                    if (self.f.tell() != self.size or len(s) != 1 or type(s[0]) is not dict
                            or self.frame_end not in (None, self.f.tell())):
                        raise ProtocolError('trailing bytes or invalid STOP/root')
                    return s[0]
                else:
                    raise ProtocolError(f'unsupported opcode 0x{op.hex()}')
                if len(s) > 256 or len(self.memo) > 512:
                    raise ProtocolError('stack/memo limit exceeded')
            raise ProtocolError('opcode budget exceeded')
        except (IndexError, KeyError, TypeError, UnicodeError, struct.error) as e:
            raise ProtocolError('malformed restricted pickle') from e


def validate_index(index, protocol):
    scalars = ('train_start_idx', 'valid_start_idx', 'test_start_idx')
    if set(index) not in (set(KEYS) | set(scalars), set(KEYS) | set(scalars) | {'y2'}):
        raise ProtocolError('unexpected/missing root keys')
    bounds = protocol['official_bounds']
    if (len(bounds) != 4 or any(type(v) is not int for v in bounds)
            or not 0 <= bounds[0] < bounds[1] < bounds[2] < bounds[3]):
        raise ProtocolError('invalid protocol time boundaries')
    if any(type(index[k]) is not int or index[k] != b for k, b in zip(scalars, bounds)):
        raise ProtocolError('source time boundaries do not match protocol')
    arrays = {k:v for k,v in index.items() if k not in scalars}
    if any(not isinstance(v, Array) or v.blob is None for v in arrays.values()):
        raise ProtocolError('only completed numeric ndarrays allowed')
    num = arrays['num_x']
    if len(num.shape) != 3 or num.shape[0] != bounds[-1]:
        raise ProtocolError('numeric panel dimensions mismatch')
    t, s, _ = num.shape
    cat = arrays['cat_x']
    if len(cat.shape) != 3 or cat.shape[:2] != (t, s) or cat.dtype.kind not in 'iu':
        raise ProtocolError('categorical panel dimensions/type mismatch')
    if num.dtype.kind != 'f' or arrays['y1'].dtype.kind != 'f':
        raise ProtocolError('floating features/labels required')
    for key in arrays.keys()-{'num_x','cat_x'}:
        if arrays[key].shape != (t,s): raise ProtocolError('mask/label dimensions mismatch')
    if any(arrays[k].dtype.kind != 'b' for k in ('mask_x', 'mask_y')):
        raise ProtocolError('boolean masks required')
    spans = sorted((v.blob.offset, v.blob.offset+v.blob.length) for v in arrays.values())
    if any(b > c for (_,b),(c,_) in zip(spans,spans[1:])):
        raise ProtocolError('aliased/overlapping arrays forbidden')
    return arrays


class LimitedReader:
    def __init__(self, stream, remaining): self.stream, self.remaining = stream, remaining
    def read(self, n=-1):
        n = min(CHUNK, self.remaining, CHUNK if n < 0 else n)
        data = self.stream.read(n); self.remaining -= len(data)
        return data


def inflate_probe(source, raw):
    """Exact observed outer wrapper only; never execute its pickle."""
    source = Path(source)
    if source.stat().st_size > MAX_SOURCE:
        raise ProtocolError('synthetic-only source cap; real conversion not enabled')
    with source.open('rb') as f:
        header = f.read(7)
        if len(header) != 7 or header[:3] != b'\x80\x05B':
            raise ProtocolError('unsupported outer wrapper')
        size = struct.unpack('<I', header[3:])[0]
        if source.stat().st_size != size+9:
            raise ProtocolError('outer payload length mismatch')
        f.seek(7+size)
        if f.read(2) != b'\x94.': raise ProtocolError('outer trailer mismatch')
        f.seek(7)
        params = zstd.get_frame_parameters(f.read(min(18, size)))
        if not 0 < params.content_size <= MAX_RAW or params.window_size > 4*CHUNK:
            raise ProtocolError('zstd size/window outside synthetic budget')
        f.seek(7)
        count = 0
        with zstd.ZstdDecompressor(max_window_size=4*CHUNK).stream_reader(
                LimitedReader(f, size), read_across_frames=True) as reader, Path(raw).open('xb') as out:
            while True:
                chunk = reader.read(CHUNK)
                if not chunk: break
                count += len(chunk)
                if count > params.content_size: raise ProtocolError('decoded size exceeds declaration')
                out.write(chunk)
            if count != params.content_size: raise ProtocolError('truncated zstd content')
            out.flush(); os.fsync(out.fileno())
    return count


def write_prefix(raw, arrays, protocol, source_hash, dest):
    """Byte copying and incremental row hashes: no full-array mmap/resident copy."""
    cutoff = protocol['official_bounds'][1]
    files, shapes, payloads = {}, {}, {}
    with Path(raw).open('rb') as stream:
        for key in KEYS:
            arr = arrays[key]
            shape = (cutoff, *arr.shape[1:]); shapes[key] = list(shape)
            row_size = math.prod(shape[1:])*arr.dtype.itemsize
            stream.seek(arr.blob.offset)
            path = dest/f'{key}.npy'
            rows_hash = hashlib.sha256()
            with path.open('xb') as out:
                np.lib.format.write_array_header_2_0(out, {'descr':arr.dtype.str,
                    'fortran_order':False, 'shape':shape})
                for _ in range(cutoff):
                    h = hashlib.sha256(canonical([shape[1:], arr.dtype.str]).encode())
                    left = row_size
                    while left:
                        block = stream.read(min(CHUNK, left))
                        if not block: raise ProtocolError('raw file changed/truncated during copy')
                        h.update(block); out.write(block); left -= len(block)
                    rows_hash.update(h.hexdigest().encode())
                out.flush(); os.fsync(out.fileno())
            payloads[key] = rows_hash.hexdigest()
            files[key] = {'shape':list(shape), 'dtype':arr.dtype.str, 'sha256':file_sha256(path)}
    axes = {key:list(map(str,range(size))) for key,size in (
        ('num_feature_ids', arrays['num_x'].shape[2]), ('cat_feature_ids', arrays['cat_x'].shape[2]),
        ('stock_ids', arrays['num_x'].shape[1]))}
    contract = {'schema':'oof-prefix-v1', 'source_sha256':source_hash, 'bounds':protocol['official_bounds'],
                'shapes':shapes, 'axes':axes, 'payloads':payloads}
    manifest = {'schema':'prefix-store-v1', 'contract':contract, 'contract_hash':digest(contract), 'files':files,
                'conversion':{'method':'restricted-protocol4-probe-v1', 'axes':'original positional order',
                    'parser_sha256':file_sha256(__file__), 'synthetic_only':True}}
    with (dest/'manifest.json').open('x', encoding='utf-8') as f:
        f.write(canonical(manifest)); f.flush(); os.fsync(f.fileno())


def verify_output(dest):
    """Read back output row hashes without mmap or full-array allocation."""
    manifest = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
    contract = manifest['contract']
    if manifest['contract_hash'] != digest(contract):
        raise ProtocolError('output contract changed')
    for key in KEYS:
        path = dest/f'{key}.npy'
        with path.open('rb') as f:
            if np.lib.format.read_magic(f) != (2,0):
                raise ProtocolError('unexpected output NPY version')
            shape, order, dtype = np.lib.format.read_array_header_2_0(f, max_header_size=4096)
            spec = manifest['files'][key]
            if list(shape) != spec['shape'] or dtype.str != spec['dtype'] or order:
                raise ProtocolError('output shape/order/dtype mismatch')
            rows_hash = hashlib.sha256()
            for _ in range(shape[0]):
                h = hashlib.sha256(canonical([shape[1:],dtype.str]).encode())
                left = math.prod(shape[1:])*dtype.itemsize
                while left:
                    block = f.read(min(CHUNK,left))
                    if not block: raise ProtocolError('output payload truncated')
                    h.update(block); left -= len(block)
                rows_hash.update(h.hexdigest().encode())
            if f.read(1) or rows_hash.hexdigest() != contract['payloads'][key]:
                raise ProtocolError('output payload mismatch')
        if file_sha256(path) != spec['sha256']:
            raise ProtocolError('output file hash mismatch')


def cleanup_owned(work, destination):
    """Remove only a tool-owned sibling workspace with a matching ownership record."""
    work, destination = Path(work).resolve(), Path(destination).resolve()
    marker = json.loads((work/'owner.json').read_text(encoding='utf-8'))
    if (work.parent != destination.parent or work == destination
            or not work.name.startswith('.numpy-probe-') or marker != {
                'work':str(work), 'destination':str(destination), 'schema':'numpy-probe-owner-v1'}
            or any(p.is_symlink() or p.is_junction() for p in [work, *work.rglob('*')])):
        raise ProtocolError('refuse cleanup outside owned workspace')
    shutil.rmtree(work)


def convert_probe(source, destination, protocol, hook=None):
    """Only small synthetic inputs; exclusive staging then same-volume publication.

    hook(stage, work) is for interruption tests, never for injecting pickle callbacks.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.stat().st_size > MAX_SOURCE:
        raise ProtocolError('synthetic-only probe; real data.z is intentionally blocked')
    if destination.exists(): raise FileExistsError(destination)
    if not destination.parent.is_dir(): raise ProtocolError('output parent must already exist')
    if shutil.disk_usage(destination.parent).free < MAX_RAW*2 + CHUNK:
        raise ProtocolError('insufficient temporary disk budget')
    work = Path(tempfile.mkdtemp(prefix='.numpy-probe-', dir=destination.parent)).resolve()
    (work/'owner.json').write_text(canonical({'work':str(work), 'destination':str(destination),
        'schema':'numpy-probe-owner-v1'}), encoding='utf-8')
    try:
        initial_hash = file_sha256(source)
        raw = work/'inner.pickle'
        raw_size = inflate_probe(source, raw)
        if hook: hook('inflated', work)
        with raw.open('rb') as f:
            arrays = validate_index(NumericIndex(f).parse(), protocol)
        package = work/'package'; package.mkdir()
        write_prefix(raw, arrays, protocol, initial_hash, package)
        if file_sha256(source) != initial_hash:
            raise ProtocolError('source changed during conversion')
        output_size = sum(p.stat().st_size for p in package.iterdir())
        if hook: hook('ready', work)
        verify_output(package)
        # Windows rename refuses an existing destination; same-volume directory publication.
        if os.name != 'nt': raise ProtocolError('atomic no-replace publication verified only on Windows')
        package.rename(destination)
        return {'raw_bytes':raw_size, 'output_bytes':output_size,
                'temporary_peak_bytes':raw_size+output_size, 'source_sha256':initial_hash}
    finally:
        cleanup_owned(work, destination)
