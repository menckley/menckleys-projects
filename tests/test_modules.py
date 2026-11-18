"""Tests unitaires du paquet `pricing`."""

import math

import numpy as np
import pytest

from pricing import binomial_tree_crr, black_scholes, monte_carlo, monte_carlo_with_error

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


def test_bsm_reference_value():
    assert black_scholes(S0, K, T, R, SIGMA, "call") == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    call = black_scholes(S0, K, T, R, SIGMA, "call", q)
    put = black_scholes(S0, K, T, R, SIGMA, "put", q)
    assert call - put == pytest.approx(S0 * math.exp(-q * T) - K * math.exp(-R * T))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_crr_converges_to_bsm(option_type):
    ref = black_scholes(S0, K, T, R, SIGMA, option_type)
    assert binomial_tree_crr(S0, K, T, R, SIGMA, 2000, option_type) == pytest.approx(ref, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_within_confidence_interval(option_type):
    ref = black_scholes(S0, K, T, R, SIGMA, option_type)
    price, err = monte_carlo_with_error(S0, K, T, R, SIGMA, 200_000, option_type, seed=42)
    assert abs(price - ref) < 4 * err


def test_monte_carlo_seed_is_reproducible():
    a = monte_carlo(S0, K, T, R, SIGMA, 1000, "call", seed=1)
    assert a == monte_carlo(S0, K, T, R, SIGMA, 1000, "call", seed=1)


def test_option_type_is_case_insensitive():
    assert black_scholes(S0, K, T, R, SIGMA, "CALL") == black_scholes(S0, K, T, R, SIGMA, "call")


@pytest.mark.parametrize(
    "call",
    [
        lambda t: black_scholes(S0, K, T, R, SIGMA, t),
        lambda t: binomial_tree_crr(S0, K, T, R, SIGMA, 10, t),
        lambda t: monte_carlo(S0, K, T, R, SIGMA, 10, t),
    ],
)
def test_invalid_option_type_raises(call):
    with pytest.raises(ValueError):
        call("straddle")


def test_prices_are_floats():
    assert isinstance(black_scholes(S0, K, T, R, SIGMA, "put"), float)
    assert np.isfinite(binomial_tree_crr(S0, K, T, R, SIGMA, 10, "put"))
