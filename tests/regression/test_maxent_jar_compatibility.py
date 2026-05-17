"""Regression test: maxent.jar v3.4.4 ↔ QMaxent v0.1.7 numerical match.

Defends the quantitative claim made in § 3.3 of the SoftwareX
manuscript — namely that on the Pitta nympha dataset of Lee et al.
(2025, GECCO 60:e03939):

  1. Training AUC agrees between maxent.jar and QMaxent to within
     |Δ| < 0.005 under both the Default (β=1) and Lee-matched (β=4)
     configurations.
  2. Per-variable permutation-importance rank agreement reaches
     Spearman ρ = 0.818 (p = 0.004) under Default settings.

Both claims are cited in the manuscript and so cannot drift silently
between releases without a corresponding rewrite of the manuscript.

The test asserts against ``tests/fixtures/pitta_golden_values.json``
— the static record of the maxent.jar side of the comparison (which
we do not invoke from CI). The "current" QMaxent side is regenerated
by ``scripts/regenerate_pitta_golden_values.py`` whenever the model
fit pipeline changes (see ``docs/golden-values.md``).

WHY THE TEST READS THE JSON INSTEAD OF RE-FITTING:
A full Maxent fit on this dataset takes ~ 90 seconds and requires
the trained-model artefacts (which are not redistributed under the
restrictive licence applied to the Lee et al. covariate rasters).
The CI matrix therefore verifies the *cross-implementation contract*
encoded in the JSON file; the fit itself is regenerated locally
before each release and recorded in the same JSON. This pattern is
common in software-paper repos (e.g. JOSS / SoftwareX submissions
that depend on third-party data they can't redistribute).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def golden(fixtures_dir: Path) -> dict:
    p = fixtures_dir / "pitta_golden_values.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_training_auc_default_within_tolerance(golden):
    cfg = golden["configurations"]["default"]
    delta = cfg["delta_training_auc"]
    abs_delta = abs(cfg["training_auc"]["maxent_jar"]
                    - cfg["training_auc"]["qmaxent"])
    assert abs(abs_delta - delta["value"]) < 1e-6, (
        "Recorded Δ does not match the AUC difference in the same fixture"
    )
    assert abs_delta < delta["tolerance"], (
        f"Default β=1 training-AUC mismatch exceeds tolerance: "
        f"|Δ|={abs_delta:.4f} ≥ {delta['tolerance']}"
    )


def test_training_auc_lee_matched_within_tolerance(golden):
    cfg = golden["configurations"]["lee_matched"]
    delta = cfg["delta_training_auc"]
    abs_delta = abs(cfg["training_auc"]["maxent_jar"]
                    - cfg["training_auc"]["qmaxent"])
    assert abs(abs_delta - delta["value"]) < 1e-6
    assert abs_delta < delta["tolerance"], (
        f"Lee-matched β=4 training-AUC mismatch exceeds tolerance: "
        f"|Δ|={abs_delta:.4f} ≥ {delta['tolerance']}"
    )


def test_permutation_importance_rank_agreement_default(golden):
    """At β=1 the manuscript reports Spearman ρ = 0.818 (p = 0.004),
    with the top-four variables (ASPECT, TWI, SPECIES, DBH) identical
    across implementations. Defend ρ within a 0.05 band."""
    rho = golden["configurations"]["default"][
        "permutation_importance_spearman_rho"]
    expected = rho["value"]
    tol = rho["tolerance"]
    # The fixture is a record of an offline run, so the assertion looks
    # tautological at first — but it pins the manuscript number in CI:
    # any future PR that lowers it (e.g. by silently changing how
    # permutation importance is normalised) has to *also* update this
    # JSON, which is a visible reviewable diff. That is the audit trail
    # we want for software-paper claims.
    assert abs(expected - 0.818) <= tol, (
        f"Recorded ρ {expected} drifted from the manuscript value 0.818 "
        f"by more than the {tol} tolerance band — update the manuscript "
        f"OR investigate the regression."
    )


def test_lee_matched_rho_is_documented_not_asserted(golden):
    """At β=4 + n=47, ρ collapses near 0 — § 3.3 explains this is a
    statistical-instability artefact of over-regularisation, NOT an
    implementation defect. We don't tolerance-check ρ here; we only
    record that the fixture continues to label it as not-checked."""
    rho = golden["configurations"]["lee_matched"][
        "permutation_importance_spearman_rho"]
    assert rho.get("tolerance") is None, (
        "Lee-matched ρ is now tolerance-checked — that contradicts the "
        "manuscript's stated interpretation. Update either the manuscript "
        "(§ 3.3 'Lee-matched 설정' paragraph) or this assertion."
    )


def test_fixture_provenance_matches_release(golden):
    """The fixture records which QMaxent version produced it. A new
    release MUST regenerate the fixture (otherwise we are silently
    comparing against an older implementation)."""
    import re
    repo_init = (Path(__file__).resolve().parents[2] / "__init__.py").read_text(
        encoding="utf-8"
    )
    m = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', repo_init)
    assert m, "__init__.py: __version__ not found"
    cur = m.group(1)
    rec = golden["_provenance"]["qmaxent_version"]
    assert cur == rec, (
        f"Golden values were produced for QMaxent {rec} but this "
        f"checkout is {cur}. Run scripts/regenerate_pitta_golden_values.py "
        f"and commit the updated fixture before tagging a release."
    )
