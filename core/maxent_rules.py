"""Maxent rules and helpers shared across the QMaxent plugin.

This module holds plain-Python helpers that encode the documented
behaviour of Maxent / maxnet and are independent of QGIS, Qt, and any
specific UI or worker. Anything that multiple modules need (the worker,
dialogs, future tests) and that does not depend on QGIS lives here.
"""


# ---------------------------------------------------------------------------
# Feature type auto-selection (maxnet logic)
# ---------------------------------------------------------------------------


def auto_feature_types(n_presence: int) -> list:
    """Select feature types based on sample size (maxnet default).

    Single source of truth for the auto-feature rule across QMaxent.

    Follows maxnet.formula(classes = "default") of the maxnet R package
    (Phillips et al. 2017). It is a default, not a recommendation of the
    optimal feature classes, which depend on the data (Merow et al.
    2013):
        n < 10  -> L         (linear only)
        n < 15  -> LQ        (+ quadratic)
        n < 80  -> LQH       (+ hinge)
        n >= 80 -> LQPH      (+ product)

    v0.2.0: threshold features are no longer added for n >= 80 (the
    maxent.jar "auto features" rule did). With elapid's scikit-learn
    backend all features share one regularization strength, and
    threshold features then overfit: on the Bradypus benchmark the
    random 10-fold test AUC was 0.873 with LQPHT vs 0.890 with LQPH
    (maxent.jar LQPHT 0.895). Users can still select threshold
    features manually.

    Args:
        n_presence: number of presence points.

    Returns:
        List of feature type strings for elapid MaxentModel.
    """
    if n_presence < 10:
        return ["linear"]
    elif n_presence < 15:
        return ["linear", "quadratic"]
    elif n_presence < 80:
        return ["linear", "quadratic", "hinge"]
    else:
        return ["linear", "quadratic", "product", "hinge"]
