# R0 experiment reliability

Scope: checkpoint restoration, verified caches, fusion score spaces and metric
validation. No default model hyperparameters were changed. No contest training,
test-label evaluation, Git staging, commit or push is part of R0.

## Checkpoints

GRU load restores the saved configuration, effective ordered columns, channels,
architecture and category sizes before constructing the network. It discards old
feature arrays. Training entrypoints load first and prepare features second, and
reject feature conflicts with the requested training configuration. Only explicit
training controls may override saved settings. A load is weight initialization,
not an exact optimizer/RNG resume.

New metadata includes feature axes/specification and a weights SHA256. Old GRU
metadata containing cfg, effective cols and category sizes remains loadable;
missing metadata is rejected. Old checkpoints without named feature axes cannot
retroactively prove semantic column identity. Callers must preserve source order.

## Caches

`load_panel` binds the source file SHA256 to the panel. Split manifests record the
source, time boundaries, complete ordered numerical/category axes, stock axis,
industry column, preprocessing version/dtypes and hashes of all cached arrays.
If named axis IDs are unavailable, positions are bound to the immutable source
hash; this is not a claim that anonymous columns have known economic meanings.

For new research, supply `expected_contract=panel_contract(data, industry_col)`
when the panel is available. For cache-only consumption, supply `data_path` (or
configure the source with JINGGE_DATA/local.yaml) to verify the source bytes.
All splits in one load must share a contract. SHA256 validation adds a sequential
source-file read; it does not decompress or train. In-memory mutations must carry
updated axis IDs/contracts; an anonymous, undeclared permutation cannot be inferred.

Legacy caches are explicitly rejected. They are not silently migrated or
overwritten. After approval, build a new directory with:

```powershell
.\.venv\Scripts\python.exe -B scripts/dump_split_cache.py --data PATH_TO_DATA --dest outputs/split_cache_r0
$env:JINGGE_SPLIT_CACHE = 'D:\友安杯资料\stock-mzz\outputs\split_cache_r0'
```

This command was NOT executed during R0. The new test split cache has features,
masks and metadata only, with neither y1 nor mask_y. `eval_rankic` accepts train
and valid only. Collapse-day statistics use separate R0 paths, bind source,
configuration, code and checkpoint hashes, and validate payload hashes on reuse.

Historical `.npy` predictions do not yet have universal provenance sidecars.
They must not be presented as fresh OOF predictions merely because cache validation
passes. A new OOF run must generate predictions from models fitted inside its folds.

## Fusion path review

| Paths | Score-space policy |
|---|---|
| FusionModel / train_fusion | raw historical blends or both branches ranked; unchanged |
| coverage/hard gate, conservative, fixed mixes, hist gates, full/ensemble evaluators | historical raw GRU/tree stage, then rank MLP stage; unchanged |
| softgate | both branches undergo the same raw/rank/industry transform before gating |
| stock-conditional / collapse switch / leftover | existing raw branch mixes; retained, scale-sensitive by design |
| climb rank ensemble | FIXED: rank-averaged GRU and tree both converted to rank before gating |

`ScorePanel` tags transformed in-memory panels; low-level blends reject mismatched
tags. `blend_in_space(..., space='rank'|'zscore')` normalizes both inputs explicitly.
Raw mode intentionally preserves historical coefficients and is NOT scale invariant.
Plain historical NPY arrays contain no score-space metadata, so their space cannot
be inferred reliably from values. Tags do not survive `np.save`/`np.load`; callers
must declare/normalize their spaces, not assume the tag is a disk schema.

Corrected climb artifacts use `*_r0_rank*` names. Historical 0.117177 and 0.122226
recipes remain unchanged. Their tests deliberately read only legacy VALID data
in a read-only audit path; this does not mark the old cache as trusted for research.

## Tests and limitations

```powershell
New-Item -ItemType Directory -Force outputs/r0_test_tmp | Out-Null
$env:TEMP = "$PWD\outputs\r0_test_tmp"
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_r0*.py' -v
.\.venv\Scripts\python.exe -B tests/test_coverage_gate.py
```

The synthetic tests include one three-day, one-epoch tiny-network smoke test;
there is no contest-data training. Historical tests skip when private valid
artifacts are unavailable. They check scores 0.11717674392133287 and
0.12222639755275264; the latter also checks exact saved-array equality.

Remaining prerequisites for time-out baselines: build verified private caches,
confirm label horizon/purge gap, freeze fold/selection protocol, and bind freshly
generated prediction artifacts to their fold/checkpoint. Exact CS-MLP optimizer
resumption and end-to-end retraining reproduction are outside this R0 change.

## Acceptance result (2026-10-31)

- 21 R0 unittest cases passed, including the two private valid historical replays.
- Existing test_coverage_gate.py passed all seven function tests.
- Existing gru_n2000_covchan_e1.pt loads successfully as 23 columns + 2 channels.
- Python AST checks and git diff --check passed.
- HEAD, index file bytes, index entries and cached diff SHA256 match the pre-edit
  snapshot. All R0 changes are unstaged/untracked; the original 88 staged files
  are intact in the index.
- No real-data cache rebuild or contest training was executed. Production entry
  requires new verified caches and a confirmed time-out evaluation protocol.
