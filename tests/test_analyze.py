import pytest
import numpy as np
from six import PY3

from visual_behavior import analyze


@pytest.mark.parametrize("kwargs, expected, exception", [
    (
        {"hit_rate": 0.8, "fa_rate": 0.2, "limits": (0.01, 0.99, ), },
        1.6832424671458286,
        None,
    ),
    (
        {"hit_rate": 0.5, "fa_rate": 0.5, "limits": (0.01, 0.99, ), },
        0.0,
        None,
    ),
    (
        {"hit_rate": 0.2, "fa_rate": 0.2, "limits": (-0.01, 0.99, ), },
        0.22,
        AssertionError,
    ),
    (
        {"hit_rate": 0.2, "fa_rate": 0.2, "limits": (0.01, 1.99, ), },
        0.22,
        AssertionError,
    ),
])
def test_dprime(kwargs, expected, exception):
    if exception is not None:
        pytest.raises(exception, analyze.dprime, **kwargs)
    else:
        assert analyze.dprime(**kwargs) == expected


@pytest.mark.skipif(PY3, reason="bug in the code, can't pass for py3")
@pytest.mark.parametrize("kwargs, expected", [
    (
        {"x": [0, 1, 2, 3, ], "time": [1, 3, 4, 5, ], },
        np.array([0.0, 0.5, 1.0, 1.0, ]),
    ),
])
def test_calc_deriv(kwargs, expected):
    np.testing.assert_equal(analyze.calc_deriv(**kwargs), expected)


@pytest.mark.parametrize("kwargs, expected", [
    ({"speed_rad_per_s": 2 * np.pi, }, 0.6035080320814271, ),
])
def test_rad_to_dist(kwargs, expected):
    assert analyze.rad_to_dist(**kwargs) == expected
