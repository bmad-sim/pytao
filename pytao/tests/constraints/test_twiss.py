import pytest

from pytao.constraints.observables.twiss import BmagTwissComparison


@pytest.mark.parametrize(
    "beta0, alpha0, beta1, alpha1, expected_pass",
    [
        (5.0, 0.5, 5.0, 0.5, True),
        (5.0, 0.5, 5.05, 0.51, True),
        (1.0, 0.0, 10.0, 0.0, False),
        (4.0, 1.0, 4.0, -1.0, False),
        (5.0, 0.0, 5.04, 0.0, True),
        (2.0, 0.3, 20.0, 0.3, False),
    ],
)
def test_bmag_twiss_comparison(beta0, alpha0, beta1, alpha1, expected_pass):
    cmp = BmagTwissComparison()
    result = cmp(beta0, alpha0, beta1, alpha1)
    assert result.passed == expected_pass
