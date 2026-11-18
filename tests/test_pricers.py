"""Cohérence des trois pricers du notebook (BSM, binomial CRR, Monte Carlo) avec q≠0."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

NOTEBOOK = Path(__file__).resolve().parent.parent / "vanillaoption.ipynb"

S0, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.03


@pytest.fixture(scope="module")
def pricers():
    """Charge uniquement les définitions de fonctions du notebook."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = {"np": np, "norm": norm}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if funcs:
            module = ast.Module(body=funcs, type_ignores=[])
            exec(compile(module, str(NOTEBOOK), "exec"), ns)  # noqa: S102 - code du notebook du dépôt
    return ns["black_scholes"], ns["binomial_tree_crr"]


def test_put_call_parity_with_dividend(pricers):
    bs, _ = pricers
    call = bs(S0, K, T, R, SIGMA, "call", Q)
    put = bs(S0, K, T, R, SIGMA, "put", Q)
    assert call - put == pytest.approx(S0 * np.exp(-Q * T) - K * np.exp(-R * T))


def test_q_zero_is_default_and_unchanged(pricers):
    bs, _ = pricers
    assert bs(S0, K, T, R, SIGMA, "call") == bs(S0, K, T, R, SIGMA, "call", 0.0)
    assert bs(S0, K, T, R, SIGMA, "call") == pytest.approx(10.4505835722, abs=1e-8)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_matches_bsm_with_dividend(pricers, option_type):
    bs, bt = pricers
    ref = bs(S0, K, T, R, SIGMA, option_type, Q)
    assert bt(S0, K, T, R, SIGMA, 500, option_type, Q) == pytest.approx(ref, abs=0.02)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_drift_matches_bsm_with_dividend(pricers, option_type):
    bs, _ = pricers
    z = np.random.default_rng(0).standard_normal(400_000)
    s_t = S0 * np.exp((R - Q - 0.5 * SIGMA**2) * T + SIGMA * np.sqrt(T) * z)
    payoff = np.maximum(s_t - K, 0) if option_type == "call" else np.maximum(K - s_t, 0)
    mc = np.exp(-R * T) * payoff.mean()
    assert mc == pytest.approx(bs(S0, K, T, R, SIGMA, option_type, Q), abs=0.1)
