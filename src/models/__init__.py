from .fusion import (
    FusionModel,
    coverage_gate_blend,
    coverage_stock_three_way,
    linear_blend,
    n2000_hard_gate_mlp,
    panel_rank_mean,
    stack_mlp_blend,
)

__all__ = [
    "LightGBMBaseline",
    "TemporalModel",
    "GRUModel",
    "CSMLPModel",
    "FusionModel",
    "linear_blend",
    "coverage_gate_blend",
    "coverage_stock_three_way",
    "n2000_hard_gate_mlp",
    "panel_rank_mean",
    "stack_mlp_blend",
]


def __getattr__(name: str):
    if name == "LightGBMBaseline":
        from .baseline import LightGBMBaseline

        return LightGBMBaseline
    if name == "TemporalModel":
        from .temporal import TemporalModel

        return TemporalModel
    if name == "GRUModel":
        from .gru_ts import GRUModel

        return GRUModel
    if name == "CSMLPModel":
        from .cs_mlp import CSMLPModel

        return CSMLPModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
