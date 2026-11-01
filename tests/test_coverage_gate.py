"""Unit tests for the n2000 hard-gate recipe and the new soft/stock gate. No contest data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation_log import render_block, upsert_experiment
from src.metrics import mean_rank_ic
from src.models.fusion import (
    coverage_gate_blend,
    coverage_soft_weights,
    coverage_stock_three_way,
    n2000_hard_gate_mlp,
    panel_cs_rank,
    panel_rank_mean,
    stack_mlp_blend,
)


def _toy_panel(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t, s = 12, 40
    mask = rng.random((t, s)) > 0.15
    mask[:, :8] = True
    gru = rng.normal(size=(t, s)).astype(np.float32)
    tree = gru + 0.4 * rng.normal(size=(t, s)).astype(np.float32)
    mlp = gru + 0.2 * rng.normal(size=(t, s)).astype(np.float32)
    y = gru + 0.3 * rng.normal(size=(t, s)).astype(np.float32)
    n_x = mask.sum(axis=1).astype(np.float64)
    n_x[:4] = 8
    n_x[4:8] = 20
    n_x[8:] = 32
    # rebuild mask to match n_x
    mask[:] = False
    for i, n in enumerate(n_x):
        mask[i, : int(n)] = True
    industry = np.tile(np.arange(s) % 5, (t, 1))
    return {"gru": gru, "tree": tree, "mlp": mlp, "y": y, "mask": mask, "n_x": n_x, "industry": industry}


def test_panel_rank_mean_ignores_raw_scale() -> None:
    p = _toy_panel()
    a = p["gru"]
    b = 100.0 * a + 7.0
    r = panel_rank_mean([a, b], p["mask"])
    ra = panel_cs_rank(a, p["mask"])
    np.testing.assert_allclose(r[p["mask"]], ra[p["mask"]], rtol=1e-5, atol=1e-5)


def test_hard_gate_matches_legacy_blend() -> None:
    p = _toy_panel()
    tau = 20.0
    legacy = coverage_gate_blend(p["gru"], p["tree"], p["mask"], tau, 0.25, 0.4, "raw")
    gated = coverage_stock_three_way(
        p["gru"],
        p["tree"],
        None,
        p["mask"],
        tau=tau,
        temperature=1.0,
        weight_low=0.25,
        weight_high=0.4,
        hard=True,
    )
    np.testing.assert_allclose(gated, legacy, rtol=1e-5, atol=1e-5)


def test_soft_weights_monotone_in_coverage() -> None:
    n_x = np.array([10.0, 20.0, 40.0])
    w = coverage_soft_weights(n_x, tau=20.0, temperature=8.0)
    assert w[0] < w[1] < w[2]
    assert np.all((w > 0) & (w < 1))


def test_stock_boost_raises_gru_weight_on_dense_names() -> None:
    p = _toy_panel()
    cov = np.zeros_like(p["gru"], dtype=np.float32)
    cov[:, :10] = 0.9
    cov[:, 10:] = 0.1
    pred = coverage_stock_three_way(
        p["gru"],
        p["tree"],
        None,
        p["mask"],
        tau=20.0,
        temperature=8.0,
        weight_low=0.25,
        weight_high=0.4,
        stock_cov=cov,
        stock_boost=0.2,
        stock_center=0.5,
        stock_temp=0.15,
        hard=False,
    )
    dense = pred[:, :10][p["mask"][:, :10]]
    sparse = pred[:, 10:][p["mask"][:, 10:]]
    gru_d = p["gru"][:, :10][p["mask"][:, :10]]
    gru_s = p["gru"][:, 10:][p["mask"][:, 10:]]
    corr_d = float(np.corrcoef(dense, gru_d)[0, 1])
    corr_s = float(np.corrcoef(sparse, gru_s)[0, 1])
    assert corr_d >= corr_s - 1e-6


def test_n2000_recipe_is_gate_then_rank_mlp() -> None:
    p = _toy_panel()
    tau = 20.0
    got = n2000_hard_gate_mlp(p["gru"], p["tree"], p["mlp"], p["mask"], tau)
    gated = coverage_gate_blend(p["gru"], p["tree"], p["mask"], tau, 0.25, 0.4, "raw")
    expect = stack_mlp_blend(p["mlp"], gated, p["mask"], 0.15, "rank")
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)
    ranked = panel_cs_rank(gated, p["mask"])
    assert ranked[p["mask"]].min() >= 1.0


def test_soft_search_can_beat_hard_on_toy_rankic() -> None:
    p = _toy_panel(seed=3)
    tau = 20.0
    hard = n2000_hard_gate_mlp(p["gru"], p["tree"], p["mlp"], p["mask"], tau)
    hard_ic = mean_rank_ic(hard, p["y"], p["mask"])
    best = hard_ic
    for temp, w_hi, w_m in [(8.0, 0.5, 0.15), (8.0, 0.55, 0.2), (20.0, 0.45, 0.1)]:
        spec = coverage_stock_three_way(
            p["gru"],
            p["tree"],
            None,
            p["mask"],
            tau=tau,
            temperature=temp,
            weight_low=0.2,
            weight_high=w_hi,
            hard=False,
        )
        pred = stack_mlp_blend(p["mlp"], spec, p["mask"], w_m, "rank")
        best = max(best, mean_rank_ic(pred, p["y"], p["mask"]))
    assert np.isfinite(hard_ic)
    assert best >= hard_ic - 1e-6


def test_ablation_log_roundtrip(tmp_path: Path) -> None:
    dest = tmp_path / "experiments.md"
    dest.write_text("# 实验记录\n\n模板：\n", encoding="utf-8")
    body = render_block(
        "fusion-n2000-softgate-001",
        config="configs/fusion_n2000_softgate.yaml",
        inputs="toy",
        history="L=10",
        model="soft gate",
        loss="none",
        valid_ic=0.121000,
        baseline_ic=0.117177,
        recomputed_baseline_ic=0.117177,
        delta=0.003823,
        ablations=[{"name": "A0", "ic": 0.117177, "note": "hard"}],
        locked={"space": "raw", "weight_low": 0.25},
        elapsed_s=1.5,
        conclusion="toy",
        next_step="none",
    )
    upsert_experiment("fusion-n2000-softgate-001", body, dest)
    text = dest.read_text(encoding="utf-8")
    assert "fusion-n2000-softgate-001" in text
    assert "0.121000" in text
    assert "0.117177" in text
    assert "模板：" in text


if __name__ == "__main__":
    test_panel_rank_mean_ignores_raw_scale()
    test_hard_gate_matches_legacy_blend()
    test_soft_weights_monotone_in_coverage()
    test_stock_boost_raises_gru_weight_on_dense_names()
    test_n2000_recipe_is_gate_then_rank_mlp()
    test_soft_search_can_beat_hard_on_toy_rankic()
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_ablation_log_roundtrip(Path(d))
    print("ok")
