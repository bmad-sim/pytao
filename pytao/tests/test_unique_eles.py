import pytest

from .. import AnyTao
from ..errors import TaoCommandError
from ..model.ele.ele import ElementID
from .conftest import packaged_examples_root
from .test_interface_commands import new_tao

WALL_INIT = "$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
FORK_INIT = str(packaged_examples_root / "fork" / "tao.init")


@pytest.fixture
def wall_tao(tao_cls: type[AnyTao]):
    with new_tao(tao_cls, init_file=WALL_INIT) as tao:
        yield tao


@pytest.fixture
def fork_tao(tao_cls: type[AnyTao]):
    with new_tao(tao_cls, init_file=FORK_INIT) as tao:
        yield tao


def test_unique_eles_superuniverse(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.unique_ele_ids() == ["1@0>>0", "1@0>>1", "1@0>>2"]


def test_unique_eles_by_index(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        # Tao locator grammar: a bare integer is an element index (branch 0).
        assert tao.unique_ele_ids("1") == ["1@0>>1"]


def test_unique_eles_universe_and_index(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        # Tao locator grammar: "1@0" is element index 0 of universe 1.
        assert tao.unique_ele_ids("1@0") == ["1@0>>0"]


def test_unique_eles_element(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.unique_ele_ids("1@0>>0") == ["1@0>>0"]


def test_unique_eles_missing(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        # Bare names are element locators; a non-matching name yields no IDs.
        assert tao.unique_ele_ids("foo") == []


def test_ele_ids_bad_universe(wall_tao: AnyTao):
    # An "@" only splits off a universe if the prefix uses universe-spec
    # characters, so a malformed-but-plausible spec is the raising case.
    with pytest.raises(TaoCommandError):
        wall_tao.ele_ids("[1:]@*")
    # "[1:foo]@*" is not a universe spec at all: it reads as an element name.
    assert wall_tao.ele_ids("[1:foo]@*") == []


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        pytest.param("1@0>>B", ["1@0>>1"], id="by-name"),
        pytest.param("0>>B", ["1@0>>1"], id="by-name-default-universe"),
        pytest.param("1@0>>*", ["1@0>>0", "1@0>>1", "1@0>>2"], id="wildcard"),
        pytest.param("1@0>>NOTFOUND", [], id="no-match"),
        pytest.param("1", ["1@0>>1"], id="by-index"),
        pytest.param("1:2", ["1@0>>1", "1@0>>2"], id="index-range"),
        pytest.param("1@2", ["1@0>>2"], id="universe-and-index"),
        pytest.param("*", ["1@0>>0", "1@0>>1", "1@0>>2"], id="wildcard-name"),
        pytest.param("1@*", ["1@0>>0", "1@0>>1", "1@0>>2"], id="universe-wildcard-name"),
        pytest.param("*@*", ["1@0>>0", "1@0>>1", "1@0>>2"], id="all-universes"),
        pytest.param("-1@*", ["1@0>>0", "1@0>>1", "1@0>>2"], id="default-universe"),
        pytest.param("[1]@*", ["1@0>>0", "1@0>>1", "1@0>>2"], id="universe-list"),
        pytest.param("B", ["1@0>>1"], id="bare-name"),
        pytest.param("B*", ["1@0>>0", "1@0>>1"], id="bare-wildcard"),  # BEGINNING, B
        pytest.param("sbend::*", ["1@0>>1"], id="bare-key-qualified"),
        pytest.param("1@sbend::B", ["1@0>>1"], id="key-qualified-with-universe"),
    ],
)
def test_ele_ids_selectors(wall_tao: AnyTao, selector: str, expected: list[str]):
    assert wall_tao.ele_ids(selector) == expected


def test_ele_ids_as_element_id(wall_tao: AnyTao):
    ids = wall_tao.ele_ids("1@0>>*", as_string=False)
    assert ids == [
        ElementID(universe=1, branch=0, ele_id="0"),
        ElementID(universe=1, branch=0, ele_id="1"),
        ElementID(universe=1, branch=0, ele_id="2"),
    ]
    assert [str(id_) for id_ in ids] == wall_tao.ele_ids("1@0>>*")
    # The no-selector (superuniverse) path honors as_string, too
    assert wall_tao.ele_ids(as_string=False) == ids


def test_ele_ids_deduplicates(wall_tao: AnyTao):
    assert wall_tao.ele_ids("0", "1@0", "1@0>>1") == ["1@0>>0", "1@0>>1"]


def test_ele_qualified_str(wall_tao: AnyTao):
    ele = wall_tao.ele("1@0>>1", defaults=False)
    assert ele.head.name == "B"
    assert ele.id.universe == 1
    assert ele.id.branch == 0


def test_ele_qualified_element_id(wall_tao: AnyTao):
    ele = wall_tao.ele(ElementID(universe=1, branch=0, ele_id="1"), defaults=False)
    assert ele.head.name == "B"


def test_eles_qualified_element_index(wall_tao: AnyTao):
    # Per Tao's locator grammar, "1@2" is element index 2 (branch 0) of universe 1.
    (ele,) = wall_tao.eles("1@2", defaults=False)
    assert ele.head.name == "END"


def test_ele_unqualified_uses_default_universe(wall_tao: AnyTao):
    ele = wall_tao.ele("B", defaults=False)
    assert ele.id.universe == wall_tao.default_universe


def test_ele_ids_fork_branch_by_name(fork_tao: AnyTao):
    expected = ["1@1>>0", "1@1>>1", "1@1>>2"]
    assert fork_tao.ele_ids("XLINE>>*") == expected
    assert fork_tao.ele_ids("1>>*") == expected


def test_ele_ids_fork_cross_branch_locator(fork_tao: AnyTao):
    assert fork_tao.ele_ids("2@1>>D2") == ["2@1>>1"]


def test_ele_ids_fork_bare_name_searches_all_branches(fork_tao: AnyTao):
    # D2 lives in branch 1; Q1 in branch 0. Both in the default universe (1).
    assert fork_tao.ele_ids("D2") == ["1@1>>1"]
    assert fork_tao.ele_ids("quad::*") == ["1@0>>3"]


def test_ele_ids_fork_universe_specs(fork_tao: AnyTao):
    assert fork_tao.ele_ids("*@D2") == ["1@1>>1", "2@1>>1"]
    assert fork_tao.ele_ids("[1:2]@D2") == ["1@1>>1", "2@1>>1"]
    assert fork_tao.ele_ids("[1,2]@D2") == ["1@1>>1", "2@1>>1"]
    # A bare wildcard name matches only the default universe.
    assert fork_tao.ele_ids("*") == fork_tao.ele_ids_from_universe(1)


def test_eles_fork_branch_wildcard(fork_tao: AnyTao):
    eles = fork_tao.eles("D2", ix_branch="*", defaults=False)
    assert [(ele.head.name, ele.head.ix_branch) for ele in eles] == [("D2", 1)]


def test_eles_fork_branch_by_name(fork_tao: AnyTao):
    names = [ele.head.name for ele in fork_tao.eles("XLINE>>*", defaults=False)]
    assert names == ["BEGINNING", "D2", "END"]


def test_default_universe(fork_tao: AnyTao):
    assert fork_tao.default_universe == 1
    fork_tao.default_universe = 2
    try:
        assert fork_tao.default_universe == 2
        assert fork_tao.ele("Q1", defaults=False).id.universe == 2
        assert fork_tao.ele_ids("-1@*") == fork_tao.ele_ids("2@*")
        assert fork_tao.ele_ids("D2") == ["2@1>>1"]
    finally:
        fork_tao.default_universe = 1


def test_eles_track_only(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init"
    ) as tao:
        default_names = {ele.head.name for ele in tao.eles("*", defaults=False)}
        tracked_names = {
            ele.head.name for ele in tao.eles("*", track_only=True, defaults=False)
        }

        assert "FF.QUA01" in default_names  # super lord
        assert "FF.QUA01" not in tracked_names
        assert "FF.QUA01#1" in tracked_names  # its super slave

        # The no-ele_id (superuniverse) path respects track_only, too
        assert {
            ele.head.name for ele in tao.eles(track_only=True, defaults=False)
        } == tracked_names
