# 实验记录

每次有效训练追加一条。没有数字的「大概好了」不算完成。对比实验一次只改窗口、特征集合、损失、模型四者之一。

**当前最强：valid mean RankIC 0.122226**（`gru-ensemble-full-s43-001`）。valid 已过 0.12。

## cs-lgbm-001
- 日期：2026-08-17
- 代码/配置：`configs/baseline.yaml`
- 输入特征：当天 raw 99 + 全市场 z-score 99 + 行业 z-score 99 + cat [0,1,2,3,4,6,7,8]；不用 cat_5
- 是否看历史：窗口 = 0（仅当天）
- 模型：LightGBM 回归，n_estimators=400, lr=0.05, num_leaves=31；训练按日最多 800 只股票
- 损失：MSE 回归 y1；验收 RankIC
- valid mean RankIC：0.099910
- train mean RankIC（如有）：未算
- 耗时 / 硬件：本机 16GB RAM，CPU torch；valid 预测文件 21:19 写出
- 结论：保留作截面底仓。超过单因子 0.079，低于门槛 0.12
- 下一步：阶段 3 看崩溃日；再加历史统计特征（提示 1）

## hist-lgbm-001
- 日期：2026-08-17
- 代码/配置：`configs/hist_lgbm.yaml`
- 输入特征：截面基线全部特征 + 过去 10 日 21 列的 mean/std/last（窗口 `[t-L, t)`）
- 是否看历史：窗口 = 10
- 模型：LightGBM 回归，超参与 cs-lgbm-001 相同
- 损失：MSE 回归 y1；验收 RankIC
- valid mean RankIC：0.084406（对照 cs-lgbm-001 为 0.099910）
- train mean RankIC（如有）：未算
- 耗时 / 硬件：产物约 22:06–22:07
- 结论：**抛弃作为主模型**。历史块占了 42% 增益，但验证集掉了约 0.016，负 RankIC 日从 49 增到 76。部分最差日略改善，整体更差。
- 下一步：已改 history.source=cs_zscore（相对强弱轨迹），重跑 hist_lgbm.yaml

## hist-lgbm-002
- 日期：2026-08-17
- 代码/配置：`configs/hist_lgbm.yaml`（`source: cs_zscore`）
- 输入特征：截面基线 + 过去 10 日、每天先截面 z-score 再 mean/std/last
- 是否看历史：窗口 = 10，`[t-L, t)`
- 模型：LightGBM 回归，超参与 cs-lgbm-001 相同
- 损失：MSE 回归 y1；验收 RankIC
- valid mean RankIC：0.104192（对照基线 0.099910，raw 历史 0.084406）
- train mean RankIC（如有）：未算
- 耗时 / 硬件：产物约 22:36
- 结论：**保留为当前最强单模型**。修好了 raw 历史的掉分；负 RankIC 日 49→39。仍低于门槛 0.12，提升约 0.004，243 天验证集上属于小幅、方向正确。
- 下一步：按日 LambdaRank（同一套特征，只改损失）

## hist-rank-001
- 日期：2026-08-17
- 代码/配置：`configs/hist_rank.yaml`
- 输入特征：与 hist-lgbm-002 相同（cs 历史）
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：LGBMRanker LambdaRank；按日 5 档相关性；num_leaves=24, max_depth=6, min_child_samples=100, lambda_l2=2.0, n_estimators=250
- 损失：lambdarank（组=交易日）
- valid mean RankIC：-0.021285（对照 hist-lgbm-002 为 0.104192）
- train mean RankIC（如有）：未算
- 耗时 / 硬件：加载 87s，摊平 56s，拟合 75s，合计约 284s
- 结论：**抛弃**。防过拟合加得太死，排序标签又压成 5 档，分数接近随机且略反。主模型仍是 hist-lgbm-002。
- 下一步：不要继续堆 Ranker 变体；可试更轻的正则或回到回归 + 别的时序特征。不要覆盖 `hist_lgbm` 产物。

## hist-lgbm-l5-001
- 日期：2026-08-17
- 代码/配置：`configs/hist_lgbm_l5.yaml`
- 输入特征：与 hist-lgbm-002 相同，只把窗口 10 改为 5
- 是否看历史：窗口 = 5，source=cs_zscore
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：0.097031（对照 L=10 为 0.104192，截面基线 0.099910）
- 耗时 / 硬件：合计约 381s
- 结论：**抛弃**。收窗口没有涨分，略低于 L=10。主模型仍是 hist-lgbm-002。
- 下一步：不要再缩窗口；主提交用 L=10 的 `task1_hist_lgbm.npy`

## fusion-001
- 日期：2026-08-18
- 代码/配置：`configs/fusion.yaml`，`scripts/train_fusion.py`
- 输入特征：不重训。时序支=`hist_lgbm` valid 0.1042；截面支=`baseline` valid 0.0999
- 是否看历史：沿用两支已有模型；融合规则只在 valid 上估
- 模型：双轴融合。试了原始分加权、截面名次加权、行业内去均值、按当天覆盖度门控
- 损失：不训练树；验收 RankIC
- valid mean RankIC：0.106185（锁定 raw_blend，时序权重 0.7）
- 耗时 / 硬件：读数约 112s，融合本身数秒
- 结论：**保留为当前最强提交** `submissions/task1_fusion.npy`。名次融合几乎同分（0.10618）；覆盖度门控、行业中性化没有明显超过简单加权。相对单模 +0.002，243 天上可能有噪声，但方向符合官方「时间维 + 股票维融合」。
- 下一步：主提交在确认 `fusion_gru_blend` 之前仍用本文件。不要为 0.002 再海搜两棵树的权重。

## hist-ms-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_ms.yaml`
- 输入特征：关掉行业 z-score；10 日 mean/std/last/delta/ewm + 5 日 last；每天 900 只
- 是否看历史：窗口 10 + 短窗 5，source=cs_zscore
- 模型：LightGBM 回归，n_estimators=500, colsample=0.7, min_child_samples=40
- 损失：MSE 回归 y1
- valid mean RankIC：0.097166（对照 hist-lgbm-002 为 0.104192，融合 0.106185）
- 结论：**抛弃**。多尺度/ewm/delta 再加树，没有超过 L=10 的简单相对强弱。不要跑 `fusion_ms`。
- 下一步：主提交仍是 `submissions/task1_fusion.npy`；时序支不要再加复杂统计

## hist-wide-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_wide.yaml`
- 输入特征：相对 hist-lgbm-002 **去掉 raw 和行业 z-score**；每天最多 2000 只；按行业分层
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：LightGBM 回归，超参与 002 相同
- valid mean RankIC：0.090571（对照 002 为 0.104192，融合 0.106185）
- 结论：**抛弃**。一次改了三样（砍 raw、加样本、分层），把 002 里约 23% 增益的 raw 砍掉了，分数掉到 0.091。不是「样本越多越好」。
- 下一步：停止改 002 的特征配方。主提交仍是 `task1_fusion.npy`。

## gru-001
- 日期：2026-08-18
- 代码/配置：`configs/gru.yaml`，`scripts/train_gru.py`
- 输入特征：21 列当天及过去的截面 z-score 窗口；不用类别、不用 raw
- 是否看历史：窗口 = 10，含当天（官方 X[t]→y1[t]）
- 模型：1 层 GRU hidden=64，每天最多 800 只，CPU
- 损失：MSE 回归 y1；验收 RankIC
- valid mean RankIC：0.086522（对照截面树 0.099910，时序树 0.104192，树融合 0.106185）
- train mean RankIC（如有）：未算
- 结论：**作为融合支保留，不作主模型**。单模弱于截面树，但与树的日均分数 Spearman 仅 0.37–0.48（两棵树之间 0.81–0.98）。在 hist 的 39 个负 RankIC 日里救回 24 天。
- 下一步：`fusion-gru-blend-001` 已落到 0.110069，用这版 MSE GRU，不要换成 cats/ic

## fusion-gru-blend-001
- 日期：2026-08-18
- 代码/配置：`configs/fusion_gru_blend.yaml`
- 输入特征：不重训。GRU 支=`gru_valid.npy` 0.0865；树融合支=`fusion_valid.npy` 0.1062
- 是否看历史：沿用已有模型
- 模型：valid 上锁定 raw_blend，GRU 权重 0.25
- 损失：不训练；验收 RankIC
- valid mean RankIC：0.110069
- 耗时 / 硬件：读数 75s
- 结论：**当前最强。** 产物 `submissions/task1_fusion_gru_blend.npy`。比树融合 +0.004。不要覆盖旧的 `task1_fusion.npy` 之前先把本文件当候选主提交。
- 下一步：后面对照实验均未超过 0.110；停止堆 GRU 变体和 LambdaRank

## gru-cats-001
- 日期：2026-08-18
- 代码/配置：`configs/gru_cats.yaml`
- 输入特征：相对 gru-001 只加 cat_1、cat_6 embedding
- 是否看历史：窗口 = 10，含当天
- 模型：同 gru-001
- 损失：MSE
- valid mean RankIC：0.086718（对照 gru-001 为 0.086522）
- 耗时 / 硬件：拟合 436s，early stop 于 epoch 6；最佳是 epoch 3 的 0.0867
- 结论：**抛弃。** 与不加类别几乎同分。类别已被树用透，再塞进 GRU 只让第 1 轮掉到 0.057，没有新信号。不要跑 `fusion_gru_cats`。

## gru-ic-001
- 日期：2026-08-18
- 代码/配置：`configs/gru_ic.yaml`
- 输入特征：与 gru-001 相同
- 是否看历史：窗口 = 10，含当天
- 模型：同 gru-001
- 损失：当天 batch 上 1−Pearson（只改损失）
- valid mean RankIC：0.090548（对照 gru-001 为 0.086522）
- 耗时 / 硬件：拟合 254s；epoch 1 就是最佳 0.0905，之后每轮 valid 下滑
- 结论：**单模略好，融合更差。** 单模 +0.004，但和树更同向，三支融合只有 0.1084，低于 MSE GRU 的 0.1101。Pearson IC 让 GRU 去抢树已经会的排序，互补性被吃掉。融合支仍用 gru-001。

## fusion-gru-ic-001
- 日期：2026-08-18
- 代码/配置：`configs/fusion_gru_ic.yaml`
- valid mean RankIC：0.108402（锁定 rank_blend，GRU 权重 0.25）
- 结论：**抛弃。** 低于 fusion-gru-blend-001 的 0.110069。

## hist-n1200-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_n1200.yaml`
- 输入特征：相对 hist-lgbm-002 关掉行业 z-score，每天 800→1200，保留 raw
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：LightGBM 回归，超参与 002 相同
- valid mean RankIC：0.103940（对照 002 为 0.104192）
- 耗时 / 硬件：合计 272s
- 结论：**抛弃。** 多抽股票没有涨分，243 天上属于噪声。每天 800 只够用。不要再加样本。

## hist-rank-mild-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_rank_mild.yaml`
- 输入特征：与 hist-lgbm-002 相同
- 是否看历史：窗口 = 10
- 模型：LGBMRanker；15 档；超参与回归版对齐（400 棵、leaves=31）
- 损失：lambdarank
- valid mean RankIC：−0.083801（对照 hist-rank-001 为 −0.021，回归 002 为 0.104192）
- 结论：**抛弃，停止 LambdaRank。** 正则放轻之后排得更反。y1 已经是截面分位数，MSE 本身就是在拟合排序；LambdaRank 优化的是 NDCG 头部品，和全市场 Spearman 不是同一目标。重要性更集中在 cat_6。不要再开 Ranker 变体。

## hist-mkt-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_mkt.yaml`
- 输入特征：相对 hist-lgbm-002 只加当天全市场覆盖度 + 21 列截面 mean/std（绝对水平）
- 是否看历史：窗口 = 10；市场状态只用当天 mask_x
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：0.063500（对照 002 为 0.104192）
- 耗时 / 硬件：合计 418s
- 结论：**抛弃。** 重要性第三就是 `mkt_coverage`，树在用覆盖度切训练期制度，验证集崩了。不要把绝对市场水平当特征。不要跑 `fusion_mkt`。

## hist-mkt-rel-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_mkt_rel.yaml`
- 输入特征：相对 002，市场状态改成相对过去 20 日 z-score；只留覆盖度 + 4 列 std
- 是否看历史：窗口 = 10；相对统计只用 `[t-20, t)`
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：0.101414（对照 002 为 0.104192，融合 GRU 版 0.110069）
- 耗时 / 硬件：合计 439s
- 结论：**抛弃。** 不再崩，但没超过 002。主提交仍是 `task1_fusion_gru_blend.npy`（0.110）。停止往树上加全市场标量。

## cs-mlp-001
- 日期：2026-08-18
- 代码/配置：`configs/cs_mlp.yaml`，`scripts/train_cs_mlp.py`
- 输入特征：当天 21 列行业内 z-score + cat_1 embedding；不用 raw、历史、cat_6
- 是否看历史：窗口 = 0
- 模型：2 层 MLP hidden=64；每天最多 800 只
- 损失：行业残差标签上的 Pearson IC；验收仍是对原始 y1 的 RankIC
- valid mean RankIC：0.091768（对照 GRU 0.0865，截面树 0.0999，树融合 0.1062）
- 耗时 / 硬件：合计 377s，CPU
- 结论：**作为第四支保留。** 单模弱于树，但与 `fusion_gru_blend` 日均 Spearman 0.681（低于 0.8）。和树/GRU 的相关约 0.58–0.61。
- 下一步：已与 0.110 做 rank_blend，见 fusion-cs-mlp-001

## fusion-cs-mlp-001
- 日期：2026-08-18
- 代码/配置：`scripts/train_cs_mlp.py` 内嵌融合（不覆盖 `task1_fusion_gru_blend.npy`）
- 输入特征：cs_mlp valid + fusion_gru_blend valid
- 模型：锁定 rank_blend，cs_mlp 权重 0.25
- valid mean RankIC：0.111264（对照 fusion_gru_blend 0.110069）
- 结论：**小幅超过 0.110，作为新候选。** 产物 `submissions/task1_fusion_cs_mlp.npy`。243 天上 +0.001 仍可能有噪声，但相关 0.68 且 0.15/0.25/0.4 三档都 ≥0.110。未改旧主文件。

## hard-resid-001
- 日期：2026-08-18
- 代码/配置：`configs/hard_resid.yaml`，`scripts/train_hard_resid.py`
- 输入特征：与 cs_mlp 相同（21 列行业 z-score + cat_1）；难日 = hist_lgbm 在 **train** 上逐日 RankIC 最低 25%（608/2432 天，阈值 0.178，不看 valid）
- 是否看历史：窗口 = 0；树分数只用来造 OLS 残差标签
- 模型：小 MLP hidden=32，dropout=0.2，weight_decay=1e-3，难日重复 2 次；约 2153 参数
- 损失：对 `y1 ~ hist_lgbm` 的当天 OLS 残差做 Pearson IC；早停看与 0.111 的 rank_blend(0.25)
- valid mean RankIC：残差单支 **-0.083400**；与 fusion_cs_mlp 日均 Spearman **-0.7129**；融回 0.111 后锁定 raw_blend 0.111266（相对 0.111264 的 2e-6，尺度淹没，不算涨）
- train 难日均值 RankIC：0.1262（易日 0.2882）——train 上「最差 25%」仍然正、且远好于 valid 崩溃日
- 耗时 / 硬件：合计 386s，CPU；树 train 预测 146s
- 结论：**抛弃。** 难日定义没有泄露 valid，但残差支学成了 0.111 的反相。说明树在 train 上排错的部分搬不到 valid；再加重采样这些天只会拟合过拟合残差。未覆盖 `task1_fusion_cs_mlp.npy`。主候选仍是 0.111264。
- 下一步：不要再从 train 残差/难日上抠同一套截面特征。过拟合更可能来自主模型容量，而不是「还有一块可学的难日残差」。

## hist-lgbm-iter-001
- 日期：2026-08-18
- 代码/配置：`scripts/eval_lgbm_iters.py`（冻结 `hist_lgbm.txt`，不重训）
- 输入特征：与 hist-lgbm-002 相同
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：同一棵 400 棵树，推理时只用前 50/100/…/400 棵
- 损失：不训练；验收 RankIC
- valid mean RankIC：50→0.1005，100→0.1015，200→0.1020，250→0.1032，300→0.1032，350→0.1034，**400→0.104192**
- train 子集（每 8 日，304 天）：50→0.154，200→0.211，400→0.247
- 耗时 / 硬件：合计 259s
- 结论：**抛弃「往回砍树」。** train 涨得比 valid 快（过拟合是真的），但 valid 仍随轮数单调上升，峰值就在 400。提早停只会把 valid 从 0.104 降到 0.102–0.103。未覆盖 002 产物。主候选仍是 0.111264。
- 下一步：不要为过拟合再减 leaves / 减 n_estimators。树还没过 valid 的峰。

## fusion-gru-cov-001
- 日期：2026-08-19
- 代码/配置：`scripts/eval_coverage_gru.py`（不重训）
- 输入特征：GRU vs 树融合 `fusion_valid`；门控只用当天 `mask_x` 股票数
- 是否看历史：沿用已有模型
- 模型：覆盖度分位 40/50/60 × GRU 权重 {0.15,0.25,0.4} × raw/rank。旧 FusionModel 门控只搜 ≥0.5 且只用 rank，这里改掉
- 损失：不训练；验收 RankIC
- valid mean RankIC：锁定 raw，tau=4546（60 分位），低覆盖 GRU 0.25、高覆盖 0.4 → **0.111078**（对照全局 0.25 为 0.110069）
- 诊断：覆盖度与树 RankIC 相关 −0.21；Q4（股票最多）树 0.058、GRU 0.100、全局 0.25 融合只有 0.074。前 8 名全是「挤的天多听 GRU」
- 耗时 / 硬件：合计 191s，主要是读 panel
- 结论：**作为融合规则保留。** 方向稳定，不是单格刷分。产物 `submissions/task1_fusion_gru_cov.npy`，未覆盖 0.110。仍低于带 cs_mlp 的 0.111264，见下条。
- 下一步：已叠 cs_mlp，见 fusion-cs-mlp-cov-001

## fusion-cs-mlp-cov-001
- 日期：2026-08-19
- 代码/配置：同上脚本内嵌融合
- 输入特征：cs_mlp + 覆盖度门控 GRU/树
- 模型：锁定 rank_blend，cs_mlp 权重 0.15
- valid mean RankIC：**0.111971**（对照 fusion_cs_mlp 0.111264，门控 GRU/树 0.111078）
- 结论：**小幅超过 0.111，作为新候选。** 产物 `submissions/task1_fusion_cs_mlp_cov.npy`。243 天上 +0.0007 仍可能有噪声，但门控方向与 Q4 诊断一致。未覆盖 `task1_fusion_cs_mlp.npy`。离 0.12 仍约 0.008。
- 下一步：不要再加密覆盖度网格。下一条若做，用 GRU 关掉当天，或把树加到 600 棵。

## hist-lgbm-n800-001
- 日期：2026-08-19
- 代码/配置：`configs/hist_lgbm_n800.yaml`，`scripts/train_baseline.py`
- 输入特征：与 hist-lgbm-002 相同
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：LightGBM 回归，只把 n_estimators 400→800
- 损失：MSE 回归 y1
- valid mean RankIC：400→0.104192（与 002 逐位对齐），600→0.103043，**800→0.104201**
- 耗时 / 硬件：合计 479s，CPU
- 结论：**抛弃。** 400 棵已经是峰；再加 400 棵只涨 9e-6，600 棵还略掉。未覆盖 `hist_lgbm` 产物。不要再加树。
- 下一步：时序树锁 400 棵。

## gru-no-today-001
- 日期：2026-08-19
- 代码/配置：`configs/gru_no_today.yaml`，`scripts/train_gru.py`
- 输入特征：与 gru-001 相同 21 列 CS z-score；只把 `include_current_day` 关掉
- 是否看历史：窗口 = 10，`[t-L, t)`，不含当天
- 模型：同 gru-001；early stop 于 epoch 6，最佳 epoch 3
- 损失：MSE
- valid mean RankIC：0.081754（对照 gru-001 为 0.086522）
- 耗时 / 硬件：拟合 591s，合计 733s，CPU
- 结论：**作为融合支保留，不作主模型。** 单模更弱，但与树融合日均 Spearman 0.425（含当天 GRU 为 0.460），与 cs_mlp 0.503（含当天为 0.608）。与含当天 GRU 相关 0.941，不是全新信号，只是少叠当天。hist 的 39 个负日仍救回 24 天。
- 下一步：按旧覆盖度门控迁移，再叠 cs_mlp；不要重搜覆盖度网格

## fusion-gru-no-today-001
- 日期：2026-08-19
- 代码/配置：`configs/fusion_gru_no_today.yaml`
- 输入特征：不重训。GRU 支=`gru_no_today` 0.0818；树融合支=`fusion_valid` 0.1062
- 模型：valid 上锁定 raw_blend，GRU 权重 0.25
- valid mean RankIC：0.110347（对照含当天 GRU 融合 0.110069）
- 结论：**噪声级。** +0.0003 不够单独立项。产物 `submissions/task1_fusion_gru_no_today.npy`，未覆盖 0.110。真正差在覆盖度 + MLP，见下条。

## fusion-cs-mlp-cov-notoday-001
- 日期：2026-08-19
- 代码/配置：覆盖度规则从 `fusion_gru_cov_lock.json` 原样迁移（tau=4546，低覆盖 0.25 / 高覆盖 0.4，raw）；cs_mlp 用原网格搜权重
- 输入特征：gru_no_today + 树融合 + cs_mlp
- 模型：覆盖度门控后锁定 rank_blend，cs_mlp 权重 0.25
- valid mean RankIC：**0.113122**（对照含当天配方 0.111971；权重 0.15 为 0.112946，同样超过旧候选）
- 诊断：门控本身 0.111615（旧 GRU 门控 0.111078），未重搜 tau
- 结论：**作为新候选。** 产物 `submissions/task1_fusion_cs_mlp_cov_notoday.npy`。未覆盖 `task1_fusion_cs_mlp_cov.npy`。243 天上 +0.001 仍可能有噪声，但 0.15/0.25 两档都超过 0.112，且相关下降与假设一致。离 0.12 仍约 0.007。
- 下一步：不要再加树、不要重搜覆盖度。下一条若做，用最近一段 train 重拟合 hist_lgbm，或 GRU 只追加列 69/73/74/42/66/58。

## hist-lgbm-recent-001
- 日期：2026-08-19
- 代码/配置：`configs/hist_lgbm_recent.yaml`
- 输入特征：与 hist-lgbm-002 相同
- 是否看历史：窗口 = 10；拟合只用 train 末 800 天（标签），历史仍可回看更早
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：0.105431（对照 002 为 0.104192）
- 耗时 / 硬件：合计 267s
- 结论：**方向对、幅度不够。** 诊断：valid 每天 4242–4724 只，train 全程 max=4239，没有任何一天达到 valid 最小覆盖。末 800 天覆盖 3433–4239，最接近，故小幅上涨。未覆盖 002。离 0.12 仍远。
- 下一步：同一 800 天把每天抽样 800→2000，让树在大池子里排

## hist-lgbm-recent-n2000-001
- 日期：2026-08-19
- 代码/配置：`configs/hist_lgbm_recent_n2000.yaml`
- 输入特征：与 recent-001 相同，只把每天训练股票 800→2000
- 是否看历史：窗口 = 10；拟合 train 末 800 天
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：0.104616（对照 recent-800 只为 0.105431，002 为 0.104192）
- 耗时 / 硬件：合计 325s
- 结论：**抛弃。** 大池子抽样没有把树送进 valid 的覆盖区间（valid 最小 4242，train 最大仍 4239）。主树仍用 002。
- 下一步：树侧这条制度差已经试过；0.113 到 0.12 的缺口还在。

## gru-no-today-recent-001
- 日期：2026-08-19
- 代码/配置：`configs/gru_no_today_recent.yaml`
- 输入特征：与 gru_no_today 相同；只把拟合改成 train 末 800 天
- 是否看历史：窗口 = 10，不含当天
- 模型：同 gru_no_today；epoch 1 最佳 0.0884，之后过拟合掉到 0.043，早停
- 损失：MSE
- valid mean RankIC：0.088440（对照全程 no-today 0.081754，含当天 gru-001 0.086522）
- 诊断：Q4 0.0968→0.1040；与树相关 0.425→0.454
- 耗时 / 硬件：拟合 172s，合计 334s
- 结论：**作为新 GRU 支保留。** 单模是目前最强 GRU。覆盖度门控迁移后 0.1152，叠原 cs_mlp 到 **0.116009**。产物 `submissions/task1_fusion_recent_gru_cov_mlp.npy`，未覆盖 0.113。离 0.12 仍约 0.004。
- 下一步：同一套 recency 重训 cs_mlp

## cs-mlp-recent-001
- 日期：2026-08-19
- 代码/配置：`configs/cs_mlp_recent.yaml`
- 输入特征：与 cs_mlp 相同；只把拟合改成 train 末 800 天
- 是否看历史：窗口 = 0
- 模型：同 cs_mlp；epoch 1 最佳
- 损失：行业残差 Pearson IC
- valid mean RankIC：0.086070（对照全程 cs_mlp 0.091768）
- 结论：**抛弃。** 截面 MLP 砍早期数据会掉分；融合仍用原来的 cs_mlp。主候选仍是 recent GRU 门控 + 旧 MLP 的 0.116009。
- 下一步：GRU recent 每天抽样 800→2000，只改这一项

## gru-no-today-recent-n2000-001
- 日期：2026-08-19
- 代码/配置：`configs/gru_no_today_recent_n2000.yaml`
- 输入特征：与 recent-001 相同；每天训练股票 800→2000
- 是否看历史：窗口 = 10，不含当天；拟合 train 末 800 天
- 模型：同 GRU；epoch 1 最佳 0.0919，之后崩到 0.027 / −0.001，早停
- 损失：MSE
- valid mean RankIC：0.091862（对照 recent-800 只 0.088440）
- 诊断：Q4 0.1040→0.1072；与树相关 0.454→0.470
- 结论：**作为当前 GRU 支。** 覆盖度门控 + 原 cs_mlp → **0.117177**。产物 `submissions/task1_fusion_recent_gru_n2000_cov_mlp.npy`，未覆盖 0.116。离 0.12 约 0.0028。
- 下一步：若继续，同一 800 天把每天抽样提到全市场（约 3500–4200），只改这一项

## gru-no-today-recent-full-001
- 日期：2026-08-24
- 代码/配置：`configs/gru_full_submit_e1.yaml + configs/fusion_full_hardgate.yaml`
- 输入特征：全市场 GRU（末 800 天、不含当天）+ 冻结 fusion 树 + 原 cs_mlp
- 是否看历史：窗口 = 10，[t-L, t)
- 模型：同 n2000 GRU；每天训练股票 = 当日全部有标签票；覆盖度硬门控 + rank MLP
- 损失：MSE；验收 RankIC
- valid mean RankIC：0.121023
- 对照基准 gru-no-today-recent-n2000-001 文档值：0.117177
- 对照基准本机复算：0.117177
- Δ vs 文档基准：0.003846
- 消融：
  - n2000_frozen_recipe: 0.117177（对照）
  - full_frozen_recipe: 0.116887（只换 GRU）
  - full_valid_grid: 0.121023（valid grid）
- 锁定：tau_q=0.7, tau=4588.4, weight_low=0.2, weight_high=0.6, mlp_weight=0.1, note=valid grid
- train mean RankIC（如有）：未算
- 耗时 / 硬件：合计 30.6s
- 结论：**作为新 GRU 支保留。** valid 0.121023，对照冻结 n2000 门控 0.117177。未覆盖 n2000 提交文件。
- 下一步：若高于上一主候选，用 submissions/task1_fusion_full_hardgate.npy 再交一次。

## gru-ensemble-full-s43-001
- 日期：2026-08-24
- 代码/配置：`configs/gru_full_seed43.yaml + configs/fusion_gru_ensemble.yaml`
- 输入特征：n2000 GRU + 全市场 seed42 + 全市场 seed43 + 冻结树 + 原 cs_mlp
- 是否看历史：窗口 = 10，[t-L, t)
- 模型：GRU 浅融合后再覆盖度硬门控；valid 锁 mix 与门控
- 损失：不重训树/MLP；验收 RankIC
- valid mean RankIC：0.122226
- 对照基准 gru-no-today-recent-n2000-001 文档值：0.121023
- 对照基准本机复算：0.117177
- Δ vs 文档基准：0.001203
- 消融：
  - n2000_frozen: 0.117177（0.117177）
  - full_grid: 0.121023（上一刀）
  - ensemble_grid: 0.122226（本刀）
- 锁定：full_mix=0.0, s43_mix=0.0, tau_q=0.7, tau=4588.4, weight_low=0.15, weight_high=0.7, mlp_weight=0.15
- train mean RankIC（如有）：未算
- 耗时 / 硬件：合计 646.1s
- 结论：**作为新候选。** valid 0.122226，对照上一刀 0.121023 / n2000 门控 0.117177。
- 下一步：用 submissions/task1_fusion_gru_ensemble.npy 再交一次；不要覆盖 0.120075 那份备份。

模板：

```text
## 实验 ID
- 日期：
- 代码/配置：
- 输入特征：
- 是否看历史：窗口 =
- 模型：
- 损失：
- valid mean RankIC：
- train mean RankIC（如有）：
- 耗时 / 硬件：
- 结论（保留 / 抛弃 / 作为融合支）：
- 下一步：
```
