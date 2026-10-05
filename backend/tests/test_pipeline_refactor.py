import pytest
from app.config import get_settings
from app.core.models import RunOptions
from app.solver.pipeline import _resolve_strategy, ResolvedStrategy


def test_resolve_strategy_defaults_when_none():
    settings = get_settings()
    strat = _resolve_strategy(None, settings)
    assert strat.use_static_blocks is False
    assert strat.dynamic_blocks is True
    assert strat.ga_level == "group"
    assert strat.group_key == "geometry"
    assert strat.post_explode_compaction is True


def test_resolve_strategy_preserves_explicit_post_explode_compaction_false():
    settings = get_settings()
    options = RunOptions(post_explode_compaction=False)
    strat = _resolve_strategy(options, settings)
    assert strat.post_explode_compaction is False
    # Defaults should still apply to unspecified fields
    assert strat.dynamic_blocks is True
    assert strat.ga_level == "group"
    assert strat.group_key == "geometry"


def test_resolve_strategy_preserves_explicit_carton_level():
    settings = get_settings()
    options = RunOptions(ga_level="carton")
    strat = _resolve_strategy(options, settings)
    assert strat.ga_level == "carton"
    assert strat.use_static_blocks is False
    assert strat.dynamic_blocks is False


def test_resolve_strategy_preserves_explicit_static_blocks():
    settings = get_settings()
    options = RunOptions(use_static_blocks=True)
    strat = _resolve_strategy(options, settings)
    assert strat.use_static_blocks is True
    assert strat.dynamic_blocks is False
    assert strat.ga_level == "unit"


def test_resolve_strategy_preserves_explicit_group_key():
    settings = get_settings()
    options = RunOptions(group_key="customer")
    strat = _resolve_strategy(options, settings)
    assert strat.group_key == "customer"
