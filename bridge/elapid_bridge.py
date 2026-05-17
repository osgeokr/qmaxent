"""Thin adapter for the elapid Python library.

Every elapid call in QMaxent goes through this module. The adapter is
intentionally minimal — it only forwards calls and exposes a single
version-compatibility check. It does NOT add functionality, change
defaults, or intercept results. If elapid's API ever shifts, this is
the one file that needs updating.

References:
    Anderson, C.B. (2023). elapid: Species distribution modeling tools
    for Python. Journal of Open Source Software, 8(84), 4930.
    https://doi.org/10.21105/joss.04930
"""

# Last elapid release verified against this plugin. Keep in sync with
# the upper bound in core.venv_manager.REQUIRED_PACKAGES.
ELAPID_TESTED_MAJOR = 1


# ---------------------------------------------------------------------------
# Version compatibility
# ---------------------------------------------------------------------------

def check_elapid_version() -> tuple:
    """Return (ok, version_string, message).

    ok is True when the installed elapid major matches the tested major.
    A False result is informational, not fatal — callers may proceed and
    log the message.
    """
    try:
        import elapid
        version = getattr(elapid, "__version__", "unknown")
    except Exception as exc:
        return False, "not installed", f"elapid not importable: {exc}"

    try:
        major = int(str(version).split(".")[0])
    except Exception:
        return True, version, f"elapid version {version} (could not parse major)"

    if major != ELAPID_TESTED_MAJOR:
        return False, version, (
            f"elapid {version} differs from tested major "
            f"{ELAPID_TESTED_MAJOR}.x — behaviour may differ."
        )
    return True, version, f"elapid {version} (verified)"


# ---------------------------------------------------------------------------
# Model construction & I/O
# ---------------------------------------------------------------------------

def make_maxent_model(**kwargs):
    """Construct an elapid MaxentModel. Forwards all kwargs verbatim.

    The plugin's call sites set the following defaults, each backed by
    an explicit reference:
        feature_types         maxnet auto rule (Phillips & Dudik 2008;
                              Phillips et al. 2017; Merow et al. 2013).
                              See core.maxent_rules.auto_feature_types.
        beta_multiplier=1.0   Tunable regularization strength
                              (Phillips & Dudik 2008;
                              Radosavljevic & Anderson 2014).
        transform="cloglog"   Cloglog complementary-log-log link
                              recommended as default by
                              Phillips et al. 2017; statistical
                              interpretation given by Elith et al. 2011.
        use_lambdas="last",   L1-penalized regularization path fitted
        n_lambdas=200         via coordinate descent
                              (Friedman, Hastie & Tibshirani 2010).
        n_hinge_features=50   maxnet's documented default (each
        n_threshold_features=50  variable contributes 2*nknots-2
                              hinge/threshold features). elapid's own
                              default is 10; QMaxent overrides to 50
                              for maxnet fidelity.
        class_weights=100     Inhomogeneous Poisson Process /
                              infinitely-weighted logistic regression
                              equivalence (Fithian & Hastie 2013).
        use_sklearn=True      Backend toggle exposed by elapid
                              (Anderson 2023). Set False to use glmnet
                              for closer maxnet fidelity.
    """
    from elapid.models import MaxentModel
    return MaxentModel(**kwargs)


def save_object(obj, path):
    """Persist a Python object via elapid's pickle wrapper."""
    import elapid
    return elapid.save_object(obj, path)


def load_object(path):
    """Load a Python object via elapid's pickle wrapper."""
    import elapid
    return elapid.load_object(path)


# ---------------------------------------------------------------------------
# Background sampling
# ---------------------------------------------------------------------------

def sample_raster(path, count: int):
    """Random spatial sample from a raster (elapid.sample_raster)."""
    import elapid
    return elapid.sample_raster(path, count=count)


def sample_bias_file(path, count: int):
    """Sample from a bias raster (elapid.sample_bias_file).

    Implements the bias-file weighting of Phillips et al. (2009) for
    correcting sample selection bias in presence-only models.
    """
    import elapid
    return elapid.sample_bias_file(path, count=count)


# ---------------------------------------------------------------------------
# Covariate annotation
# ---------------------------------------------------------------------------

def annotate(points, raster_paths, **kwargs):
    """Extract raster values at point locations (elapid.annotate).

    Forwards every keyword argument verbatim to elapid (e.g. labels,
    drop_na, quiet) so the adapter never has to track elapid's API
    changes for this function.
    """
    import elapid
    return elapid.annotate(points, raster_paths, **kwargs)


# ---------------------------------------------------------------------------
# Spatial cross-validation strategies
# ---------------------------------------------------------------------------

def make_geographic_kfold(n_splits: int):
    """Spatial block CV via KMeans clusters (Roberts et al. 2017)."""
    from elapid.train_test_split import GeographicKFold
    return GeographicKFold(n_splits=n_splits)


def checkerboard_split(points, grid_size: float):
    """Single spatial train/test split (Muscarella et al. 2014, ENMeval)."""
    from elapid.train_test_split import checkerboard_split as _cb
    return _cb(points, grid_size=grid_size)


def make_buffered_loo(distance: float):
    """Buffered leave-one-out CV (Pearson 2007; Ploton et al. 2020)."""
    from elapid.train_test_split import BufferedLeaveOneOut
    return BufferedLeaveOneOut(distance=distance)


# ---------------------------------------------------------------------------
# Spatial projection
# ---------------------------------------------------------------------------


def apply_model_to_rasters(model, raster_paths, output_path, **kwargs):
    """Apply a fitted model to a stack of rasters (elapid.apply_model_to_rasters).

    Forwards every keyword argument verbatim. The callable
    `elapid.geo.tqdm` is intentionally NOT exposed through this adapter:
    the projection worker monkey-patches it for progress reporting and
    that requires direct access to the elapid.geo module (see
    workers.projection_worker for the documented exception).
    """
    import elapid
    return elapid.apply_model_to_rasters(
        model=model,
        raster_paths=raster_paths,
        output_path=output_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Permutation importance (added v0.1.3)
# ---------------------------------------------------------------------------

def permutation_importance_scores(model, x, y, n_repeats: int = 10,
                                  n_jobs: int = 1, random_state=None):
    """Compute permutation importance for each feature.

    Thin wrapper around elapid.MaxentModel.permutation_importance_scores,
    which itself wraps sklearn.inspection.permutation_importance. The
    metric is the AUC drop when each feature's values are randomly
    shuffled and the model re-applied. Returns an array of shape
    (n_features, n_repeats) so the caller can report mean and std.

    This is the same metric reported by maxent.jar as "permutation
    importance" in its HTML output, normalized as percentage of total
    (Phillips et al. 2006, 2017). Unlike maxent.jar's percent
    contribution (which tracks lambda accumulation during training),
    permutation importance is computed on the *trained* model and
    therefore answers a different question: "how much does the model's
    output depend on each feature" rather than "how much did each
    feature contribute to learning".

    References:
        Phillips, S.J., Anderson, R.P., Dudik, M., Schapire, R.E., &
            Blair, M.E. (2017). Opening the black box: an open-source
            release of Maxent. Ecography, 40, 887-893.
        Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T.
            (2007). Bias in random forest variable importance measures:
            illustrations, sources and a solution. BMC Bioinformatics,
            8:25 — caveats on permutation importance under collinearity.

    Args:
        model: a fitted elapid MaxentModel.
        x: design matrix used as the evaluation set (n_samples, n_features).
            For an honest estimate, pass a held-out subset (e.g. the
            test fold from CV); for a within-sample estimate, pass the
            full training matrix.
        y: target labels matching x.
        n_repeats: number of shuffles per feature (sklearn default 10).
        n_jobs: parallel jobs; default 1 to avoid nested-parallel issues
            inside Qt worker threads.
        random_state: optional int for reproducibility.

    Returns:
        importances: numpy array of shape (n_features, n_repeats).
            Each entry is the AUC drop for one feature on one shuffle.
    """
    # elapid's SDMMixin.permutation_importance_scores does not accept
    # random_state, but the underlying sklearn function does. We use
    # sklearn directly so QMaxent runs are reproducible when the user
    # has set a random seed in the ② Parameters tab.
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(
        model, x, y,
        n_repeats=n_repeats,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return pi.importances
