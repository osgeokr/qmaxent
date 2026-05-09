# References

QMaxent's defaults, evaluation procedures, and survey-planning workflows are
grounded in the species distribution modeling (SDM) literature. This chapter
groups the works the plugin draws on by topic, with a one-line annotation on
how each reference is used inside QMaxent. The plugin source code carries
inline citations to these works in the relevant modules
(`workers/maxent_worker.py`, `bridge/elapid_bridge.py`,
`bridge/priority_sites.py`).

## Maxent — core methodology

Phillips, S. J., Anderson, R. P., & Schapire, R. E. (2006). Maximum entropy
modeling of species geographic distributions. *Ecological Modelling*, 190(3–4),
231–259. <https://doi.org/10.1016/j.ecolmodel.2005.03.026>
:   The original Maxent paper. Defines the linear, quadratic, hinge, product,
    and threshold features (LQPHT), the regularization framework, and the
    interpretation of the output as a relative probability of presence.

Phillips, S. J., & Dudík, M. (2008). Modeling of species distributions with
Maxent: new extensions and a comprehensive evaluation. *Ecography*, 31(2),
161–175. <https://doi.org/10.1111/j.0906-7590.2008.5203.x>
:   Introduces the regularization-multiplier scheme that QMaxent's
    "Regularization" parameter exposes, and benchmarks Maxent against
    alternative SDM methods.

Phillips, S. J., Anderson, R. P., Dudík, M., Schapire, R. E., & Blair, M. E.
(2017). Opening the black box: an open-source release of Maxent.
*Ecography*, 40(7), 887–893. <https://doi.org/10.1111/ecog.03049>
:   The release that defines the cloglog output transform now considered the
    standard reporting format. QMaxent uses cloglog as the default in the
    Spatial Projection sub-tab.

Elith, J., Phillips, S. J., Hastie, T., Dudík, M., Chee, Y. E., & Yates, C. J.
(2011). A statistical explanation of MaxEnt for ecologists. *Diversity and
Distributions*, 17(1), 43–57.
<https://doi.org/10.1111/j.1472-4642.2010.00725.x>
:   Re-derives Maxent as a penalized log-linear (Poisson) model, providing
    the theoretical basis for the maxnet implementation that elapid (and
    therefore QMaxent) uses under the hood.

Merow, C., Smith, M. J., & Silander, J. A. (2013). A practical guide to
MaxEnt for modeling species' distributions: what it does, and why inputs
and settings matter. *Ecography*, 36(10), 1058–1069.
<https://doi.org/10.1111/j.1600-0587.2013.07872.x>
:   The standard practitioner's guide. QMaxent's recommendations on
    background sampling, feature complexity, and regularization mirror this
    paper.

Fithian, W., & Hastie, T. (2013). Finite-sample equivalence in statistical
models for presence-only data. *Annals of Applied Statistics*, 7(4),
1917–1939. <https://doi.org/10.1214/13-AOAS667>
:   Proves the equivalence between Maxent and a downweighted Poisson
    regression on a sufficient sample of background points — the basis for
    the maxnet algorithm.

## Sample selection bias and background points

Phillips, S. J., Dudík, M., Elith, J., Graham, C. H., Lehmann, A., Leathwick,
J., & Ferrier, S. (2009). Sample selection bias and presence-only
distribution models: implications for background and pseudo-absence data.
*Ecological Applications*, 19(1), 181–197.
<https://doi.org/10.1890/07-2153.1>
:   Justifies QMaxent's "Down-weight spatially clustered points" option and
    the recommended practice of drawing background points from the same
    region as presence points.

## Spatial cross-validation

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J.,
Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data
with temporal, spatial, hierarchical, or phylogenetic structure.
*Ecography*, 40(8), 913–929. <https://doi.org/10.1111/ecog.02881>
:   The conceptual foundation for QMaxent's Geographic K-Fold and Buffered
    Leave-One-Out methods. Demonstrates that random K-fold is
    over-optimistic when data are spatially autocorrelated.

Muscarella, R., Galante, P. J., Soley-Guardia, M., Boria, R. A., Kass, J. M.,
Uriarte, M., & Anderson, R. P. (2014). ENMeval: an R package for conducting
spatially independent evaluations and estimating optimal model complexity
for Maxent ecological niche models. *Methods in Ecology and Evolution*,
5(11), 1198–1205. <https://doi.org/10.1111/2041-210X.12261>
:   Source of QMaxent's Checkerboard partitioning method.

Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019).
blockCV: an R package for generating spatially or environmentally separated
folds for k-fold cross-validation of species distribution models. *Methods
in Ecology and Evolution*, 10(2), 225–232.
<https://doi.org/10.1111/2041-210X.13107>
:   Companion software reference; QMaxent's spatial-block partitioning
    follows the same logic in Python.

Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N., Picard, N.,
Rossi, V., et al. (2020). Spatial validation reveals poor predictive
performance of large-scale ecological mapping models. *Nature
Communications*, 11, 4540. <https://doi.org/10.1038/s41467-020-18321-y>
:   Empirical evidence for the gap between random-CV and spatial-CV
    performance estimates. Underpins QMaxent's default of pooled AUC over
    Buffered LOO folds.

## Model complexity, regularization, and evaluation

Radosavljevic, A., & Anderson, R. P. (2014). Making better Maxent models of
species distributions: complexity, overfitting and evaluation. *Journal of
Biogeography*, 41(4), 629–643. <https://doi.org/10.1111/jbi.12227>
:   Provides the rationale for QMaxent's auto-rule for feature selection
    based on sample size, and the default regularization multiplier of 1.0.

Lobo, J. M., Jiménez-Valverde, A., & Real, R. (2008). AUC: a misleading
measure of the performance of predictive distribution models. *Global
Ecology and Biogeography*, 17(2), 145–151.
<https://doi.org/10.1111/j.1466-8238.2007.00358.x>
:   Cautionary reference. QMaxent reports AUC alongside response curves and
    jackknife importance so users do not rely on AUC alone.

## Threshold methods and small-sample evaluation

Liu, C., White, M., & Newell, G. (2013). Selecting thresholds for the
prediction of species occurrence with presence-only data. *Journal of
Biogeography*, 40(4), 778–789. <https://doi.org/10.1111/jbi.12058>
:   Source of the MaxSSS (Maximum Sum of Sensitivity and Specificity)
    threshold offered in QMaxent's Priority Sites tab, and the comparative
    discussion of MTP and T10.

Pearson, R. G., Raxworthy, C. J., Nakamura, M., & Peterson, A. T. (2007).
Predicting species distributions from small numbers of occurrence records:
a test case using cryptic geckos in Madagascar. *Journal of Biogeography*,
34(1), 102–117. <https://doi.org/10.1111/j.1365-2699.2006.01594.x>
:   Defines the Buffered Leave-One-Out (LOO) workflow that QMaxent
    implements for small datasets (≤ 25 presences).

## Survey planning with SDM output

Williams, J. N., Seo, C., Thorne, J., Nelson, J. K., Erwin, S., O'Brien, J. M.,
& Schwartz, M. W. (2009). Using species distribution models to predict new
occurrences for rare plants. *Diversity and Distributions*, 15(4),
565–576. <https://doi.org/10.1111/j.1472-4642.2009.00567.x>
:   Conceptual basis for QMaxent's Discovery mode in the Priority Sites
    tab — using model-predicted high-suitability areas to direct new field
    surveys.

Rhoden, C. M., Peterman, W. E., & Taylor, C. A. (2017). Maxent-directed
field surveys identify new populations of narrowly endemic habitat
specialists. *PeerJ*, 5, e3632. <https://doi.org/10.7717/peerj.3632>
:   Source of QMaxent's Validation mode — stratified sampling across four
    suitability quartiles to test the prediction in the field.

## Software dependencies

Anderson, C. B. (2023). elapid: species distribution modeling tools for
Python. *Journal of Open Source Software*, 8(84), 4930.
<https://doi.org/10.21105/joss.04930>
:   The Python library that QMaxent integrates. Provides the maxnet engine,
    spatial-CV partitioners, and projection utilities.

Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
generalized linear models via coordinate descent. *Journal of Statistical
Software*, 33(1), 1–22. <https://doi.org/10.18637/jss.v033.i01>
:   The glmnet algorithm that the maxnet R package (and elapid's Python
    port) is built on.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
O., et al. (2011). Scikit-learn: machine learning in Python. *Journal of
Machine Learning Research*, 12, 2825–2830.
<https://www.jmlr.org/papers/v12/pedregosa11a.html>
:   Provides the cross-validation primitives, ROC computation, and metric
    utilities used by QMaxent's evaluation pipeline.

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen,
P., Cournapeau, D., et al. (2020). Array programming with NumPy. *Nature*,
585(7825), 357–362. <https://doi.org/10.1038/s41586-020-2649-2>
:   Numerical foundation for all of the above.

## Related SDM platforms

Kass, J. M., Vilela, B., Aiello-Lammens, M. E., Muscarella, R., Merow, C., &
Anderson, R. P. (2018). Wallace: a flexible platform for reproducible
modeling of species niches and distributions built for community
expansion. *Methods in Ecology and Evolution*, 9(4), 1151–1156.
<https://doi.org/10.1111/2041-210X.12945>
:   Cited in the Introduction as a related platform. Wallace is a
    Shiny-based GUI for SDM; QMaxent occupies a similar niche but inside
    QGIS rather than as a standalone web app.

## Case study reference

Lee, S., Cho, M., Yu, B.-H., Lee, S., Lee, S., Wolfe, J. D., & Oh, H.-S.
(2025). Breeding habitat prediction and nest-site characteristics of the
fairy pitta (*Pitta nympha*) in Geoje-si, South Korea: insights from a
species distribution model.
:   The published study reproduced in the *Pitta nympha* worked example.
    Originally analysed with the classical Java MaxEnt; the example chapter
    repeats the analysis in QMaxent and discusses agreement and divergence.
