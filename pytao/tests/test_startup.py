import pytest

from ..tao_ctypes.core import TaoInitializationError
from .. import Tao, TaoStartup


def test_examples_can_init(tao_example: TaoStartup) -> None:
    assert tao_example.can_initialize


def test_regression_tests_can_init(tao_regression_test: TaoStartup) -> None:
    assert tao_regression_test.can_initialize


@pytest.mark.parametrize(
    ("startup"),
    [
        pytest.param(TaoStartup("-i foo")),
        pytest.param(TaoStartup("-init foo")),
        pytest.param(TaoStartup("-init_file foo")),
        pytest.param(TaoStartup("-la foo")),
        pytest.param(TaoStartup("-lat foo")),
        pytest.param(TaoStartup("-lattice_file foo")),
    ],
)
def test_can_init(startup: TaoStartup) -> None:
    assert startup.can_initialize


@pytest.mark.parametrize(
    ("startup"),
    [
        pytest.param(TaoStartup()),
        pytest.param(TaoStartup(external_plotting=True)),
        pytest.param(TaoStartup(var_file="no")),
    ],
)
def test_no_init(startup: TaoStartup) -> None:
    assert not startup.can_initialize


def test_plotting() -> None:
    assert (
        TaoStartup(plot="mpl", external_plotting=True, noplot=True).tao_init
        == "-external_plotting -noplot"
    )


def test_init_override() -> None:
    assert TaoStartup("-init_file foo", init_file="test").tao_init == "-init_file foo"


def test_geometry() -> None:
    assert TaoStartup(geometry="3x3").tao_init == "-geometry 3x3"
    assert TaoStartup(geometry=(32, 23)).tao_init == "-geometry 32x23"


def test_startup_requires_init() -> None:
    with pytest.raises(TaoInitializationError):
        Tao()
    with pytest.raises(TaoInitializationError):
        Tao("bad_init")


def test_startup_bad_file() -> None:
    bad_fn = "/tmp/foooooobarrrr"
    for check in range(1, 3):
        print(f"** Check number {check} **")
        # We do this *twice* because it's actually two different paths to
        # initialization of the underlying Fortran library.
        with pytest.raises(TaoInitializationError) as raises_context:
            Tao(f"-init_file {bad_fn}")
        ex = raises_context.value
        assert hasattr(ex, "tao_output")
        assert bad_fn in ex.tao_output
        assert "TAO INITIALIZATION FILE NOT FOUND" in ex.tao_output
        print(str(ex))


fodo_lattice = """
no_digested
parameter[geometry] = closed
parameter[particle] = electron

beginning[e_tot] = 150e6
beginning[x_position]     = 1e-3
beginning[y_position]     = 1e-3
beginning[z_position]     = 0.

L_tot = 2
Lq = 0.1
Ld = (L_tot - 2*Lq)/2

k1_optimal =  (1/Lq) * 2*sqrt(2) / L_tot

K1: kicker, L=0.1

P0: pipe

P1: P0, L = Ld

q0: quad, L = Lq
q1: q0
q2: q0

O_L: overlay = {p1[L]:(Lcell - 2*Lq)/2}, var = {Lcell}
O_L[Lcell] = L_tot

q1[k1] = k1_optimal
q2[k1] = -k1_optimal

fodo1: line = (k1, p1, q1, p1, q2 )

use, fodo1
"""


def test_startup_from_lattice_contents() -> None:
    tao = Tao.from_lattice_contents(fodo_lattice)
    names = [tao.ele_head(ele_id)["name"] for ele_id in tao.unique_ele_ids()]
    assert names == [
        "BEGINNING",
        "K1",
        "O_L",
        "P1",
        "Q1",
        "P1",
        "Q2",
        "END",
    ]
