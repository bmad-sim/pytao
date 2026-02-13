import pytest

from pytao.model.ele import ElementID, ElementIntersection, ElementList, ElementRange


@pytest.mark.parametrize(
    "expected, tao_representation",
    [
        pytest.param(
            ElementID(
                ele_id="q1",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "q1",
        ),
        pytest.param(
            ElementID(
                ele_id="q1",
                key="quadrupole",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "quadrupole::q1",
        ),
        pytest.param(
            ElementID(
                ele_id="q1",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=1,
                negated=False,
            ),
            "q1+1",
        ),
        pytest.param(
            ElementID(
                ele_id="q1",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=-1,
                negated=False,
            ),
            "q1-1",
        ),
        pytest.param(
            ElementID(
                ele_id="Q3",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=True,
            ),
            "~Q3",
        ),
        pytest.param(
            ElementID(
                ele_id="45",
                key=None,
                universe=None,
                branch=2,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "2>>45",
        ),
        pytest.param(
            ElementID(
                ele_id="45:51",
                key=None,
                universe=None,
                branch=2,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "2>>45:51",
        ),
        pytest.param(
            ElementID(
                ele_id="q*",
                key="quad",
                universe=None,
                branch="x_br",
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "x_br>>quad::q*",
        ),
        pytest.param(
            ElementID(
                ele_id="q1:q5",
                key="sbend",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "sbend::q1:q5",
        ),
        pytest.param(
            ElementID(
                ele_id="a*",
                key="marker",
                universe=None,
                branch=None,
                match_number=2,
                match_offset=None,
                negated=False,
            ),
            "marker::a*##2",
        ),
        pytest.param(
            ElementID(
                ele_id="bpm*",
                key="type",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "type::bpm*",
        ),
        pytest.param(
            ElementID(
                ele_id="my duck",
                key="alias",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "alias::'my duck'",
        ),
        pytest.param(
            ElementID(
                ele_id="test",
                key=None,
                universe=None,
                branch=None,
                match_number=1,
                match_offset=2,
                negated=False,
            ),
            "test##1+2",
        ),
        pytest.param(
            ElementID(
                ele_id="test",
                key=None,
                universe=None,
                branch=None,
                match_number=1,
                match_offset=-2,
                negated=False,
            ),
            "test##1-2",
        ),
        # From bmad regression_tests/bookkeeper_test loc_str patterns:
        pytest.param(
            ElementID(
                ele_id="qu1",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=-1,
                negated=False,
            ),
            "qu1-1",
        ),
        pytest.param(
            ElementID(
                ele_id="qu1",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=-5,
                negated=False,
            ),
            "qu1-5",
        ),
        pytest.param(
            ElementID(
                ele_id="qu2",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=1,
                negated=False,
            ),
            "qu2+1",
        ),
        pytest.param(
            ElementID(
                ele_id="qu2",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=10,
                negated=False,
            ),
            "qu2+10",
        ),
        pytest.param(
            ElementID(
                ele_id="sb",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "sb",
        ),
        pytest.param(
            ElementID(
                ele_id="*",
                key="quad",
                universe=None,
                branch=1,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "1>>quad::*",
        ),
        pytest.param(
            ElementID(
                ele_id="sb",
                key=None,
                universe=None,
                branch=None,
                match_number=2,
                match_offset=None,
                negated=False,
            ),
            "sb##2",
        ),
        pytest.param(
            ElementID(
                ele_id="*",
                key="type",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "type::*",
        ),
        pytest.param(
            ElementID(
                ele_id="sb%",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "sb%",
        ),
        pytest.param(
            ElementID(
                ele_id="'q*t'",
                key="alias",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "alias::'q*t'",
        ),
        pytest.param(
            ElementID(
                ele_id="'So Long'",
                key="descrip",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "descrip::'So Long'",
        ),
        # Universe + negation (correct Tao syntax: uni@ before ~)
        pytest.param(
            ElementID(
                ele_id="Q*",
                key=None,
                universe=1,
                branch=0,
                match_number=None,
                match_offset=None,
                negated=True,
            ),
            "1@~0>>Q*",
        ),
    ],
)
def test_element_identifier_from_tao(
    expected: ElementID,
    tao_representation: str,
):
    result = ElementID.from_tao_any(tao_representation)
    assert result == expected

    assert not isinstance(result, ElementIntersection)
    assert not isinstance(result, list)

    assert (
        result.tao_string == tao_representation
    ), f"{result}.tao_string = {result.tao_string!r}, expected {tao_representation!r}"


@pytest.mark.parametrize(
    "expected, tao_representation",
    [
        pytest.param(
            ElementID(
                ele_id="ele_id",
                key="key",
                universe=None,
                branch="branch",
                match_number=1,
                match_offset=None,
                negated=False,
            ),
            "key::branch>>ele_id##1",
        ),
        pytest.param(
            ElementID(
                ele_id="ele_id",
                key=None,
                universe=None,
                branch="branch",
                match_number=1,
                match_offset=None,
                negated=False,
            ),
            "branch>>ele_id##1",
        ),
        pytest.param(
            ElementID(
                ele_id="ele_id",
                key=None,
                universe=None,
                branch=None,
                match_number=1,
                match_offset=None,
                negated=False,
            ),
            "ele_id##1",
        ),
        pytest.param(
            ElementID(
                ele_id="ele_id",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "ele_id",
        ),
        # From bmad regression_tests/bookkeeper_test loc_str:
        pytest.param(
            ElementID(
                ele_id="*",
                key="octupole",
                universe=None,
                branch=1,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "octupole::1>>*",
        ),
    ],
)
def test_element_old_style_identifier_from_tao(
    expected: ElementID,
    tao_representation: str,
):
    result = ElementID.from_tao(tao_representation)
    assert result == expected
    result = ElementID.from_tao_old_syntax(tao_representation)
    assert result == expected


# From bmad regression_tests/bookkeeper_test loc_str patterns:
@pytest.mark.parametrize(
    "start, end, tao_representation",
    [
        pytest.param(
            ElementID(ele_id="3"),
            ElementID(ele_id="15"),
            "3:15",
        ),
        pytest.param(
            ElementID(ele_id="3", branch=1, key="drift"),
            ElementID(ele_id="15", branch=1, key="drift"),
            "1>>drift::3:15",
        ),
        pytest.param(
            ElementID(ele_id="qu1", branch=0, key="drift"),
            ElementID(ele_id="qu2", branch=0, key="drift"),
            "0>>drift::qu1:qu2",
        ),
        pytest.param(
            ElementID(ele_id="qu1", branch=1, key="drift"),
            ElementID(ele_id="qu2", branch=1, key="drift"),
            "1>>drift::qu1:qu2",
        ),
        pytest.param(
            ElementID(ele_id="17", key="sbend"),
            ElementID(ele_id="5", key="sbend"),
            "sbend::17:5",
        ),
    ],
)
def test_element_range(
    start: ElementID,
    end: ElementID,
    tao_representation: str,
):
    result = ElementID.from_tao_any(tao_representation, split_range=True)
    assert isinstance(result, ElementRange)

    assert result.start == start
    assert result.end == end
    assert (
        result.tao_string == tao_representation
    ), f"{result}.tao_string = {result.tao_string!r}, expected {tao_representation!r}"


@pytest.mark.parametrize(
    "id1, id2, tao_representation",
    [
        pytest.param(
            ElementID(
                ele_id="1:10",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            ElementID(
                ele_id="BPM*",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "1:10 & BPM*",
        ),
        pytest.param(
            ElementID(
                ele_id="bpm*",
                key="type",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            ElementID(
                ele_id="my duck",
                key="alias",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "type::bpm* & alias::'my duck'",
        ),
        pytest.param(
            ElementID(
                ele_id="*",
                key="quadrupole",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            ElementID(
                ele_id="q3",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=True,
            ),
            "quadrupole::* & ~q3",
        ),
        # From bmad regression_tests/bookkeeper_test loc_str:
        pytest.param(
            ElementID(
                ele_id="*",
                key="Quad",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            ElementID(
                ele_id="*9*",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "Quad::* & *9*",
        ),
    ],
)
def test_element_intersection(
    id1: ElementID,
    id2: ElementID,
    tao_representation: str,
):
    result = ElementID.from_tao_any(tao_representation)
    assert isinstance(result, ElementIntersection)

    assert result.elements[0] == id1
    assert result.elements[1] == id2
    assert (
        result.tao_string == tao_representation
    ), f"{result}.tao_string = {result.tao_string!r}, expected {tao_representation!r}"


@pytest.mark.parametrize(
    "id1, id2, tao_representation",
    [
        pytest.param(
            ElementID(
                ele_id="1:10",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            ElementID(
                ele_id="BPM*",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "1:10, BPM*",
        ),
        pytest.param(
            ElementID(
                ele_id="bpm*",
                key="type",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            ElementID(
                ele_id="my duck",
                key="alias",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            "type::bpm*, alias::'my duck'",
        ),
        pytest.param(
            ElementID(
                ele_id="*",
                key="quadrupole",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            ElementID(
                ele_id="q3",
                key=None,
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=True,
            ),
            "quadrupole::*, ~q3",
        ),
        # From bmad regression_tests/bookkeeper_test loc_str:
        pytest.param(
            ElementID(
                ele_id="*",
                key="quad",
                universe=None,
                branch=None,
                match_number=None,
                match_offset=None,
                negated=False,
            ),
            ElementID(
                ele_id="*",
                key=None,
                universe=None,
                branch=2,
                match_number=None,
                match_offset=None,
                negated=True,
            ),
            "quad::*, ~2>>*",
        ),
    ],
)
def test_element_list(
    id1: ElementID,
    id2: ElementID,
    tao_representation: str,
):
    result = ElementID.from_tao_any(tao_representation)
    assert isinstance(result, ElementList)

    assert result.elements[0] == id1
    assert result.elements[1] == id2
    assert (
        result.tao_string == tao_representation
    ), f"{result}.tao_string = {result.tao_string!r}, expected {tao_representation!r}"
