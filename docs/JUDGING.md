# 靖戈赛道评分对照

对照《附件1》赛题设置与《附件2》靖戈命题材料。课题 **1（y1）**。评委按四块打分；本仓库每一块都有对应入口，避免只交一个 npy。

## 官方任务（附件2）

- 输入：`num_x (T,S,Nn)`、`cat_x (T,S,Nc)`，按时间切分 train / valid / test
- 指标：每个交易日在股票维 S 上算 RankIC（Spearman），再对时间取均值
- 门槛：测试集 mean RankIC **≥ 0.12**（结果质量的硬线）
- 提交：`np.float32` 数组 `(Tt, S)`，`Tt = T - test_start_idx`，本数据为 `(442, 5282)`；`np.load(path)` 无额外参数可读
- 官方提示：
  1. 在时间维 **T** 上用时序模型
  2. 在股票维 **S** 上建模，并与时序结果融合

禁止：测试集调参、把 `y2` 当 `y1` 特征、窗口看到当天之后、把 `data.z` 打进 git。

## 分数怎么拿满

| 维度 | 权重 | 评委要看到什么 | 本仓库怎么交 |
|---|---|---|---|
| 结果质量 | 40% | 测试 RankIC ≥ 0.12 | 本地只报 valid。当前硬门控 **0.117177**。主路径：T 支 GRU n2000 + S 支树/行业 MLP + 覆盖度门控。软门控是冲 0.12 的独立实验，不覆盖基线 yaml |
| 设计质量 | 20% | 结构清楚、可复现 | `src/` 库、`scripts/` 入口、`configs/` 实验。一条命令：`scripts/run_pipeline.py --list`。切分只按官方时间索引 |
| 体验性 | 20% | 能跑、别爆内存、别拖太久 | 16GB 禁止并行两份面板；融合走 `split_cache`；深度模型只训末 800 天；提交前 `check_submission.py` |
| 创新性 | 20% | 针对「横截面排序 + 覆盖度漂移」，不是换网络名字 | ① 因果截面 z-score 窗口 GRU（不含当天，避免与树重复用同一天）② 行业残差 MLP 补树的组截距 ③ **日覆盖度门控**：股票多的日子树崩、GRU 更稳，高覆盖提高时序权重 ④ 个股历史上市率软门控（`fusion_n2000_softgate`） |

## 系统怎么接在一起

```
data.z（仓库外）
    → dump_split_cache          # 标签/mask，融合不再读面板
    → baseline LightGBM (S, 当天)
    → hist_lgbm (S, 过去 10 日 cs_zscore)
    → fusion 双树（valid 锁权重）
    → cs_mlp（行业残差，S）
    → gru_no_today_recent_n2000（T）
    → coverage / softgate 融合（T+S）
    → check_submission → 上传 satc.jingle.ai
```

一条线：

```text
.\.venv\Scripts\python.exe scripts\run_pipeline.py --list
.\.venv\Scripts\python.exe scripts\run_pipeline.py --from cache --to check
.\.venv\Scripts\python.exe scripts\run_pipeline.py --stage gru_n2000 --run
```

默认 dry-run，只打印命令和缺文件。确认内存空闲后再 `--run`。

## 当前数字（valid，禁止拿 test 选模）

| 实验 | 轴 | valid RankIC | 角色 |
|---|---|---|---|
| cs-lgbm-001 | S | 0.099910 | 截面底仓 |
| hist-lgbm-002 | S | 0.104192 | 主树 |
| gru-no-today-recent-n2000-001 | T | 0.091862 | 主 GRU |
| 硬覆盖度门控 + 原 cs_mlp | T+S | **0.117177** | 当前最强已锁产物 |
| 官方门槛 | test | 0.12 | 未到；差约 0.0028 |

冲门槛：先跑 `eval_softgate_lift.py`（不重训）。若仍低于 0.12，同一 800 天把 GRU 每天抽样提到全市场，只改新 yaml，不要改 `gru_no_today_recent_n2000.yaml`。

## 创新点（写进答辩稿用这段）

横截面 RankIC 在「当天可交易股票数」变化时会塌。验证集上树与覆盖度负相关，高覆盖四分位树约 0.058、GRU 约 0.100。因此融合权重不能全局常数：覆盖高的日子加时序权重，覆盖低的日子留截面树；再按个股过去窗口的上市密度微调，避免新票被 GRU 空窗带偏。这是赛题提示 1+2 的具体化，不是再叠一层 Transformer。
