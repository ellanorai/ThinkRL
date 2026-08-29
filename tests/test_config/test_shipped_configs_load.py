"""Every non-empty YAML shipped under configs/ must load through the real loader.

The directory previously held seventeen files, sixteen of them zero bytes, and the one with
content used a flat schema the loader rejects, so no shipped config could be loaded at all.
"""

from pathlib import Path

import pytest
import yaml

from thinkrl.config.base import ThinkRLConfig


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
YAML_FILES = sorted(CONFIG_ROOT.rglob("*.yaml"))

# Files still empty because they would configure code that does not exist yet. Each entry
# should be removed as the corresponding feature lands; the test below fails if one is
# filled in and left listed here, so the list cannot rot.
BLOCKED_ON_MISSING_FEATURES = {
    "eval_config.yaml",                  # thinkrl/evaluation is four empty files
    "reasoning/cot_config.yaml",         # thinkrl/reasoning/cot is empty
    "reasoning/tot_config.yaml",         # thinkrl/reasoning/tot is empty
    "models/multimodal_config.yaml",     # PAPO is not exported, no multimodal trainer
    "models/moe_config.yaml",            # no MoE-specific handling in the model loader
    "models/t5_config.yaml",             # the algorithms assume a decoder-only causal LM
}

POPULATED = [p for p in YAML_FILES if str(p.relative_to(CONFIG_ROOT)) not in BLOCKED_ON_MISSING_FEATURES]


def _relative(path):
    return str(path.relative_to(CONFIG_ROOT))


def test_most_shipped_configs_are_populated():
    assert POPULATED, f"no populated YAML files found under {CONFIG_ROOT}"
    assert len(POPULATED) > len(BLOCKED_ON_MISSING_FEATURES)


@pytest.mark.parametrize("path", POPULATED, ids=_relative)
def test_shipped_config_is_not_a_zero_byte_file(path):
    assert path.stat().st_size > 0, f"{path.name} is empty, so it cannot configure anything"


@pytest.mark.parametrize("path", POPULATED, ids=_relative)
def test_shipped_config_loads_through_the_real_loader(path):
    assert ThinkRLConfig.from_yaml(path) is not None


@pytest.mark.parametrize("path", POPULATED, ids=_relative)
def test_shipped_config_uses_known_sections(path):
    known = {"model", "algorithm", "distributed", "data", "logging", "peft"}
    data = yaml.safe_load(path.read_text())

    assert isinstance(data, dict), f"{path.name} did not parse into a mapping"
    unknown = set(data) - known
    assert not unknown, f"{path.name} has sections the loader ignores: {sorted(unknown)}"


@pytest.mark.parametrize("name", sorted(BLOCKED_ON_MISSING_FEATURES))
def test_blocked_list_does_not_go_stale(name):
    """If a listed file gets content, it belongs in the tested set instead of on this list."""
    path = CONFIG_ROOT / name
    if not path.exists():
        pytest.skip(f"{name} has been removed")
    assert path.stat().st_size == 0, f"{name} now has content; remove it from BLOCKED_ON_MISSING_FEATURES"
