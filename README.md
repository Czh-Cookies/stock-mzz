# stock-mzz · 靖戈命题一（y1）

南京大学友安杯 · 靖戈企业命题赛道 · 课题 1。面板横截面排序，指标是 **mean RankIC**（先在股票维 S 上 Spearman，再对时间取均值）。官方测试集门槛 **0.12**。

评分怎么对齐、系统怎么接：先读 [`docs/JUDGING.md`](docs/JUDGING.md)。任务清单见 [`计划.md`](计划.md)，数字见 [`experiments.md`](experiments.md)。

当前最强本地 valid **0.117177**（末 800 天 GRU 每天 2000 只 + 硬覆盖度门控 + 原行业 MLP）。差门槛约 0.0028。测试集不参与选模。主产物不进 git：`submissions/task1_fusion_recent_gru_n2000_cov_mlp.npy`。软门控是独立实验，不覆盖这份基线。

不要把 `configs/fusion_gru_blend.yaml`（0.110）或更早融合当主方案。

## 架构（对应官方两条提示）

```
T 轴 GRU（轨迹，不含当天）     S 轴树 + 行业 MLP（当天截面）
              \                       /
               覆盖度门控 / 软门控融合
                         |
                   (Tt, S) float32 提交
```

| 层 | 目录 | 作用 |
|---|---|---|
| 库 | `src/` | 读数、因果特征、RankIC、GRU / 树 / MLP / 融合、提交校验 |
| 入口 | `scripts/` | 单阶段训练与评估；总控 `run_pipeline.py` |
| 实验 | `configs/` | 一份 yaml 一次实验；基线 yaml 不改 |
| 对照 | `docs/JUDGING.md` | 结果 40% / 设计 20% / 体验 20% / 创新 20% |

16GB 机器：**禁止并行两份全量面板**。融合阶段只读 `outputs/split_cache/`。

## 一条命令复现

```text
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe scripts\run_pipeline.py --list
.\.venv\Scripts\python.exe scripts\run_pipeline.py --scorecard
.\.venv\Scripts\python.exe scripts\run_pipeline.py --from cache --to check
```

默认 dry-run。确认没有第二份 `data.z` 在内存后再加 `--run`。单阶段：

```text
.\.venv\Scripts\python.exe scripts\run_pipeline.py --stage gru_n2000 --run
.\.venv\Scripts\python.exe scripts\run_pipeline.py --stage softgate --run
.\.venv\Scripts\python.exe scripts\check_submission.py --pred submissions\task1_fusion_n2000_softgate.npy
```

冲 0.12：先跑软门控（不重训）。不够再开新 yaml 把每天抽样提到全市场，**不要改** `configs/gru_no_today_recent_n2000.yaml`。

## 队友接手

1. 数据放仓库外（官方 `data.z`）。不要再解包 8.5GB pickle。
2. 路径写 `configs/local.yaml`（gitignore）或环境变量 `JINGGE_DATA`：

```yaml
data:
  path: "C:/path/to/data.z"
  unpacked: false
```

3. 预测、checkpoint、提交 npy **不进 git**。复现 0.117 见 `计划.md` 第 10 节与 `experiments.md` 的 `gru-no-today-recent-n2000-001`。
4. 禁止提交：`data.z`、解包 pickle、`outputs/`、`checkpoints/`、`submissions/`、`*.npy`。

已抛弃的实验不要重跑，结论在 `experiments.md`。
