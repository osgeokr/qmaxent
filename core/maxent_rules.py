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
    """Select feature types based on sample size (mirrors maxnet R package).

    Single source of truth for the auto-feature rule across QMaxent.

    Follows the documented maxnet auto rule (Phillips & Dudik 2008,
    Phillips et al. 2017, Merow et al. 2013):
        n < 10  -> L         (linear only)
        n < 15  -> LQ        (+ quadratic)
        n < 80  -> LQH       (+ hinge)
        n >= 80 -> LQPHT     (+ product, threshold)

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
        return ["linear", "quadratic", "product", "hinge", "threshold"]
