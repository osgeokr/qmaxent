"""Unit tests for the Maxent feature auto-selection rule.

Defends the boundaries documented in ``core/maxent_rules.py``, which
mirror the maxnet R-package rule (Phillips & Dudík 2008; Phillips
et al. 2017; Merow et al. 2013):

    n  <  10  →ㅤL
    10 ≤ n < 15 → LQ
    15 ≤ n < 80 → LQH
    n  ≥ 80   → LQPHT

These breakpoints are NOT free parameters — they appear verbatim in
the maxnet sources and are quoted as the auto-rule in § 2.2 (②
Parameters) of the SoftwareX manuscript. Any silent change to them
would (a) break maxent.jar comparability claimed in § 2.3 and (b)
invalidate the auto-selected configurations used in the example
Bradypus / Ariolimax demonstrations.
"""

from __future__ import annotations

import pytest

from core.maxent_rules import auto_feature_types


# Each row: (n_presence, expected list of feature types).
# Boundary rows are picked precisely at the documented breakpoints
# (n=9 vs 10, n=14 vs 15, n=79 vs 80) so any future off-by-one error
# in the rule lights up immediately.
@pytest.mark.parametrize(
    "n,expected",
    [
        (1,    ["linear"]),
        (9,    ["linear"]),
        (10,   ["linear", "quadratic"]),
        (14,   ["linear", "quadratic"]),
        (15,   ["linear", "quadratic", "hinge"]),
        (47,   ["linear", "quadratic", "hinge"]),         # § 3.3 Pitta nympha
        (79,   ["linear", "quadratic", "hinge"]),
        (80,   ["linear", "quadratic", "product", "hinge", "threshold"]),
        (116,  ["linear", "quadratic", "product", "hinge", "threshold"]),  # Bradypus
        (5000, ["linear", "quadratic", "product", "hinge", "threshold"]),
    ],
)
def test_auto_feature_types_breakpoints(n, expected):
    assert auto_feature_types(n) == expected


def test_returns_a_new_list_each_call():
    # Defensive: callers may mutate the returned list (e.g. append a
    # custom feature). The rule must hand out independent lists, not
    # a shared module-level singleton, or one call will leak features
    # into the next.
    a = auto_feature_types(50)
    b = auto_feature_types(50)
    a.append("threshold")
    assert "threshold" not in b


def test_only_documented_feature_strings_emitted():
    # The set of legal feature-type strings is fixed by elapid's
    # MaxentModel; emitting anything outside this set would crash the
    # actual fit at runtime with an unhelpful error far from this
    # source location. Catch it here.
    valid = {"linear", "quadratic", "hinge", "product", "threshold"}
    for n in (0, 5, 10, 15, 50, 80, 1000):
        types = set(auto_feature_types(n))
        assert types <= valid, f"unexpected feature(s) at n={n}: {types - valid}"
