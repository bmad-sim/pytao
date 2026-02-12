import pytest

from pytao.model.ele import ElementID, ElementIntersection, ElementList


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
