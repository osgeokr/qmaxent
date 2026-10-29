"""Regression test: maxent.jar v3.4.4 vs QMaxent numerical agreement.

Defends the quantitative claims of § 3.3 of the accompanying manuscript
(v0.2.0) on the Pitta nympha dataset of Lee et al.
(2025, GECCO 64:e03939):

  1. On identical inputs (same presences and background, LQH, full
     data) training AUC agrees within 0.02 and cloglog predictions
     correlate at r >= 0.85, at beta 1 and beta 4.
  2. Permutation-importance rankings, both evaluated on the training
     data, agree strongly at beta 4 (Spearman rho >= 0.8).
  3. Under the Lee et al. final-model protocol the QMaxent ranking
     agrees with Lee et al. Table 4 (rho >= 0.75).

The test asserts against ``tests/fixtures/pitta_golden_values.json``,
the record of an offline run (the Lee et al. rasters cannot be
redistributed; see docs/golden-values.md). The point of the test is an
audit trail: a PR that changes any of these numbers must also change
the JSON, which is a visible, reviewable diff.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def golden(fixtures_dir: Path) -> dict:
    p = fixtures_dir / "pitta_golden_values.json"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("config", ["default_beta1", "lee_settings_beta4"])
def test_training_auc_within_tolerance(golden, config):
    cfg = golden["implementation_equivalence"][config]
    auc = cfg["training_auc"]
    abs_delta = abs(auc["maxent_jar"] - auc["qmaxent"])
    delta = cfg["delta_training_auc"]
    assert abs(abs_delta - delta["value"]) < 1e-3, "Recorded delta does not match the AUCs"
    assert abs_delta < delta["tolerance"], (
        f"{config}: training-AUC difference {abs_delta:.4f} >= {delta['tolerance']}"
    )


@pytest.mark.parametrize("config", ["default_beta1", "lee_settings_beta4"])
def test_predictions_agree(golden, config):
    r = golden["implementation_equivalence"][config]["prediction_pearson_r"]
    assert r["value"] >= r["minimum"], f"{config}: prediction r {r['value']} < {r['minimum']}"


@pytest.mark.parametrize("config", ["default_beta1", "lee_settings_beta4"])
def test_permutation_rank_agreement(golden, config):
    rho = golden["implementation_equivalence"][config]["permutation_importance_spearman_rho"]
    assert rho["value"] >= rho["minimum"], (
        f"{config}: rank agreement rho {rho['value']} fell below {rho['minimum']}; "
        f"update the manuscript (§ 3.3) or investigate the regression."
    )


def test_lee_protocol_matches_published_ranking(golden):
    rho = golden["lee_protocol"]["permutation_importance_spearman_rho"]["lee2025_vs_qmaxent"]
    assert rho["value"] >= rho["minimum"]
    assert golden["lee_protocol"]["top3_all"] == ["TWI", "ASPECT", "SPECIES"]


def test_fixture_covariates_are_lee_table1(golden):
    """The covariate list must match Lee et al. (2025, Table 1)."""
    assert golden["dataset"]["covariates"] == [
        "AGE",
        "ASPECT",
        "CANOPY_COVER",
        "DBH",
        "HEIGHT",
        "SLOPE",
        "SMI",
        "SPECIES",
        "TIN",
        "TWI",
    ]


def test_fixture_provenance_matches_release(golden):
    """A new release must regenerate the fixture (otherwise we would
    silently compare against an older implementation)."""
    repo_init = (Path(__file__).resolve().parents[2] / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', repo_init)
    assert m, "__init__.py: __version__ not found"
    assert golden["_provenance"]["qmaxent_version"] == m.group(1), (
        "Golden values were produced for a different QMaxent version; "
        "regenerate them (docs/golden-values.md) before tagging a release."
    )
