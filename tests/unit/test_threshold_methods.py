"""Unit tests for the canonical SDM threshold methods.

Validates ``bridge.priority_sites.compute_threshold`` for the four
methods documented in the manuscript (§ 2.2 ⑤, Table referencing Liu
2013 and Pearson 2007):

  - MTP    — minimum training presence
  - T10    — 10th percentile of training-presence predictions
  - MaxSSS — threshold that maximises sensitivity + specificity
  - Custom — user-supplied numeric value

These methods feed the Priority Sites for Survey workflow that is one
of QMaxent's three named contributions (closing the "post-decision
gap" in Table 1 of the manuscript). Silent regressions here would
change which raster cells become candidate field sites in real
conservation deployments.
"""

from __future__ import annotations

import numpy as np
import pytest

from bridge.priority_sites import compute_threshold


def test_mtp_returns_minimum_of_training_predictions():
    tp = [0.31, 0.42, 0.97, 0.55, 0.68]
    assert compute_threshold("MTP", tp) == pytest.approx(0.31)


def test_t10_uses_linear_percentile():
    # numpy's default percentile uses linear interpolation between the
    # two nearest training-presence values. With 11 evenly spaced
    # values [0.0, 0.1, …, 1.0] the 10th percentile is exactly 0.1.
    tp = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert compute_threshold("T10", tp) == pytest.approx(0.1)


def test_maxsss_picks_threshold_that_separates_two_distributions():
    # Presences cluster high, background clusters low. MaxSSS should
    # pick a cut between the two — verified here by checking that the
    # chosen threshold (a) is above every background value and (b) is
    # at-or-below every presence value, the textbook 100 % sens +
    # 100 % spec outcome of Liu (2013).
    rng = np.random.default_rng(0)
    tp = rng.uniform(0.7, 1.0, size=200).tolist()
    bg = rng.uniform(0.0, 0.3, size=200).tolist()
    t = compute_threshold("MaxSSS", tp, background_predictions=bg)
    assert max(bg) < t <= min(tp)


def test_maxsss_requires_background():
    with pytest.raises(ValueError, match="background"):
        compute_threshold("MaxSSS", [0.1, 0.2])


def test_custom_threshold_passthrough():
    assert compute_threshold("Custom", [0.1, 0.2], custom_value=0.42) == 0.42


def test_custom_requires_value():
    with pytest.raises(ValueError, match="custom_value"):
        compute_threshold("Custom", [0.1, 0.2])


def test_empty_training_predictions_raises():
    with pytest.raises(ValueError, match="No training predictions"):
        compute_threshold("MTP", [])


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown threshold method"):
        compute_threshold("NotAThing", [0.1, 0.2])
