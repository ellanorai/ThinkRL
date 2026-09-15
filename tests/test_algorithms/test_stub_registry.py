"""KTO, ORPO and RLOO are registered and exported as though they were available (#76).

All three carry `TODO: Implement algorithm` and raise from `__init__` itself, and their
create_* factories exist only to raise. Anyone reading `ALGORITHMS` or the package
namespace sees sixteen algorithms, three of which cannot be constructed.
"""

import pytest

from thinkrl.algorithms import ALGORITHMS, get_algorithm, is_stub, list_algorithms


STUBS = {"kto", "orpo", "rloo"}


def test_the_three_stubs_are_identified():
    detected = {name for name, cls in ALGORITHMS.items() if is_stub(cls)}

    assert detected == STUBS


@pytest.mark.parametrize("name", sorted(STUBS))
def test_get_algorithm_refuses_a_stub_at_lookup(name):
    """Rejected here rather than several frames later inside __init__."""
    with pytest.raises(NotImplementedError, match="registered but not implemented"):
        get_algorithm(name)


@pytest.mark.parametrize("name", sorted(STUBS))
def test_the_refusal_points_at_what_does_work(name):
    with pytest.raises(NotImplementedError, match="grpo"):
        get_algorithm(name)


def test_get_algorithm_still_returns_implemented_algorithms():
    """The guard must not make the registry useless for the ones that work."""
    from thinkrl.algorithms import GRPOAlgorithm

    assert get_algorithm("grpo") is GRPOAlgorithm


def test_unknown_names_still_raise_value_error():
    """A typo and an unimplemented algorithm are different failures."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        get_algorithm("not_an_algorithm")


def test_list_algorithms_can_exclude_the_stubs():
    assert STUBS.issubset(set(list_algorithms()))
    assert not STUBS & set(list_algorithms(include_stubs=False))


def test_the_cli_uses_this_definition_rather_than_its_own():
    """thinkrl/cli/main.py kept a second copy of the same logic; two answers to "which
    algorithms are real" is one more than the codebase can keep consistent."""
    from thinkrl.cli.main import _is_stub

    assert _is_stub is is_stub
