"""Smoke regression test: the elapid Bradypus example fits and projects.

Runs the canonical Bradypus variegatus example (§ 3.1 of the
manuscript) end-to-end through the *pure-Python* elapid backend that
QMaxent depends on. The test is a smoke test, not a tight numerical
test: its only job is to fail loudly when a future elapid /
scikit-learn / numpy upgrade breaks the basic fit+predict contract on
the manuscript's flagship demo dataset.

Skipped automatically when ``elapid`` is not available in the test
environment (e.g. the lint-only matrix row); see ``conftest.py``.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.needs_elapid, pytest.mark.needs_sklearn]


def test_bradypus_default_fit_produces_finite_auc():
    """Fit elapid's MaxentModel on the bundled Bradypus dataset and
    verify the training AUC is in (0.5, 1.0). Catches the broad class
    of upstream regressions where the fit returns NaN / -inf / 1.0
    exactly, all of which would invalidate the § 3.1 demonstration."""
    elapid = pytest.importorskip("elapid")
    np = pytest.importorskip("numpy")
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    # elapid ships the Bradypus sample as a downloadable resource; use
    # the in-package loader if present, otherwise skip — the test is
    # responsible for failing on real regressions, not on missing
    # sample data.
    try:
        from elapid.datasets import load_sample_data
    except ImportError:
        pytest.skip("elapid.datasets.load_sample_data not available")

    try:
        sample = load_sample_data("bradypus")
    except Exception as e:
        pytest.skip(f"Could not load bradypus sample: {e}")

    # The elapid sample loader's exact return shape varies across
    # releases; we accept either ``(presence_df, background_df)``
    # or a single combined DataFrame with a ``label`` column.
    if isinstance(sample, tuple) and len(sample) == 2:
        x_pres, x_bg = sample
        X = np.vstack([x_pres.values, x_bg.values])
        y = np.concatenate([
            np.ones(len(x_pres), dtype=int),
            np.zeros(len(x_bg), dtype=int),
        ])
    else:
        df = sample
        label_col = next(
            (c for c in df.columns if c.lower() in ("label", "presence")),
            None,
        )
        if label_col is None:
            pytest.skip("bradypus sample has no recognisable label column")
        y = df[label_col].values.astype(int)
        X = df.drop(columns=[label_col]).values

    model = elapid.MaxentModel()
    model.fit(X, y)
    proba = model.predict(X)
    auc = sklearn_metrics.roc_auc_score(y, proba)

    assert np.isfinite(auc), f"Bradypus training AUC is not finite: {auc!r}"
    # Generous band — purpose is to detect collapse to chance or to
    # perfect fit, not to police the exact AUC. The full numerical
    # claim is defended by the Pitta golden-values test.
    assert 0.55 < auc < 0.999, (
        f"Bradypus training AUC out of plausible range: {auc:.4f}"
    )
