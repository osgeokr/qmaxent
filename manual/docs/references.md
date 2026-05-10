# References

QMaxent's defaults, evaluation procedures, and survey-planning workflows are
grounded in the species distribution modeling (SDM) literature. This chapter
groups the works the plugin draws on by topic, with a one-line annotation on
how each reference is used inside QMaxent. The plugin source code carries
inline citations to these works in the relevant modules
(`workers/maxent_worker.py`, `bridge/elapid_bridge.py`,
`bridge/priority_sites.py`).

## Maxent — core methodology

**Phillips, S. J., Anderson, R. P., & Schapire, R. E. (2006).** Maximum
entropy modeling of species geographic distributions. *Ecological
Modelling*, 190(3-4), 231–259.
*The original Maxent paper. QMaxent's feature-class taxonomy (LQPHT),
jackknife importance, and ROC-based evaluation all come from this
formulation.*

**Phillips, S. J., & Dudík, M. (2008).** Modeling of species
distributions with Maxent: new extensions and a comprehensive evaluation.
*Ecography*, 31(2), 161–175.
*The benchmark paper that established the sample-size feature-class rule
QMaxent's "Auto" mode follows, and the regularization-multiplier
default of 1.0.*

**Phillips, S. J., Anderson, R. P., Dudík, M., Schapire, R. E., &
Blair, M. E. (2017).** Opening the black box: an open-source release of
Maxent. *Ecography*, 40(7), 887–893.
*The release of the open-source Maxent and the formal recommendation
to adopt the cloglog output transform — adopted as QMaxent's default.*

**Fithian, W., & Hastie, T. (2013).** Finite-sample equivalence in
statistical models for presence-only data. *The Annals of Applied
Statistics*, 7(4), 1917–1939.
*The paper that establishes the formal equivalence between Maxent and
infinitely-weighted logistic regression — the foundation for elapid's
implementation.*

## elapid — software backend

**Anderson, C. B. (2023).** elapid: Species distribution modeling
tools for Python. *Journal of Open Source Software*, 8(84), 5435.
*The peer-reviewed release of the elapid library that QMaxent uses for
its `MaxentModel`, `GeographicKFold`, and feature transformers.*

**Pedregosa, F., et al. (2011).** Scikit-learn: Machine learning in
Python. *Journal of Machine Learning Research*, 12, 2825–2830.
*The numerical core that elapid's regularized-likelihood maximisation
delegates to.*

**Harris, C. R., et al. (2020).** Array programming with NumPy.
*Nature*, 585, 357–362.
*The array library underlying elapid, scikit-learn, and the QMaxent
raster-extraction layer.*

## Cross-validation and model evaluation

**Roberts, D. R., et al. (2017).** Cross-validation strategies for data
with temporal, spatial, hierarchical, or phylogenetic structure.
*Ecography*, 40(8), 913–929.
*The paper that established **spatial cross-validation as the default**
for SDMs — directly shaping QMaxent's Geographic K-Fold default and
the warning text in [The Parameters tab](parameters-tab.md).*

**Hijmans, R. J. (2012).** Cross-validation of species distribution
models: removing spatial sorting bias and calibration with a null
model. *Ecology*, 93(3), 679–688.
*Provides the formal argument for buffered leave-one-out CV when
sample sizes are small; QMaxent's *Buffered LOO* option implements this.*

**Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G.
(2019).** blockCV: An R package for generating spatially or
environmentally separated folds for k-fold cross-validation of species
distribution models. *Methods in Ecology and Evolution*, 10(2), 225–232.
*The block-CV recipes implemented by QMaxent's Checkerboard option.*

**Muscarella, R., et al. (2014).** ENMeval: An R package for conducting
spatially independent evaluations and estimating optimal model
complexity for Maxent ecological niche models. *Methods in Ecology and
Evolution*, 5(11), 1198–1205.
*The hyperparameter-tuning framework whose checkerboard partitioning is
implemented as a QMaxent CV option.*

**Kass, J. M., et al. (2018).** Wallace: A flexible platform for
reproducible modeling of species niches and distributions built for
community expansion. *Methods in Ecology and Evolution*, 9(4),
1151–1156.
*The interactive ENM platform whose hyperparameter-tuning workflow QMaxent
intentionally complements rather than duplicates.*

## Sample bias correction

**Phillips, S. J., et al. (2009).** Sample selection bias and
presence-only distribution models: implications for background and
pseudo-absence data. *Ecological Applications*, 19(1), 181–197.
*The paper underlying QMaxent's *Down-weight spatially clustered points*
option in the Parameters → Advanced section.*

**Boria, R. A., Olson, L. E., Goodman, S. M., & Anderson, R. P.
(2014).** Spatial filtering to reduce sampling bias can improve the
performance of ecological niche models. *Ecological Modelling*, 275,
73–77.
*Cited in [Data tab](data-tab.md) as the reason to spatially-thin
presence layers before modeling.*

**Elith, J., Kearney, M., & Phillips, S. (2010).** The art of modelling
range-shifting species. *Methods in Ecology and Evolution*, 1(4),
330–342.
*The MESS-style extrapolation analysis whose logic underlies QMaxent's
unified projection-preflight dialog.*

## SDM best-practice reviews

**Elith, J., et al. (2011).** A statistical explanation of MaxEnt for
ecologists. *Diversity and Distributions*, 17(1), 43–57.
*The standard pedagogical reference for understanding what Maxent does
and does not assume.*

**Merow, C., Smith, M. J., & Silander Jr, J. A. (2013).** A practical
guide to MaxEnt for modeling species' distributions: what it does, and
why inputs and settings matter. *Ecography*, 36(10), 1058–1069.
*The companion practical guide to Elith et al. 2011.*

**Araújo, M. B., et al. (2019).** Standards for distribution models in
biodiversity assessments. *Science Advances*, 5(1), eaat4858.
*The IPBES-aligned reporting standard QMaxent's XLSX export was designed
to satisfy. Cited throughout this manual.*

**Radosavljevic, A., & Anderson, R. P. (2014).** Making better Maxent
models of species distributions: complexity, overfitting and evaluation.
*Journal of Biogeography*, 41(4), 629–643.
*The empirical paper documenting how feature-class tuning without
spatial CV inflates AUC; underlies the warning text in the
[Parameters tab](parameters-tab.md) about the Auto rule.*

## Survey design

**Williams, J. N., et al. (2009).** Using species distribution models
to predict new occurrences for rare plants. *Diversity and
Distributions*, 15(4), 565–576.
*The original *Discovery*-mode survey-design paper whose workflow QMaxent
implements in the Priority Sites tab.*

**Rhoden, C. M., Peterman, W. E., & Taylor, C. A. (2017).** Maxent-
directed field surveys identify new populations of narrowly endemic
habitat specialists. *PeerJ*, 5, e3632.
*The model-validation companion to Williams et al. 2009; QMaxent's
*Model validation* mode reproduces this design.*

**Stevens, D. L. Jr., & Olsen, A. R. (2004).** Spatially balanced
sampling of natural resources. *Journal of the American Statistical
Association*, 99(465), 262–278.
*The spatially-balanced-sampling theory underlying QMaxent's
between-candidate spacing constraint.*

**Robinson, N. M., et al. (2018).** Refining survey effort: how
detection probability shapes the design of biodiversity surveys.
*Methods in Ecology and Evolution*, 9(3), 575–587.
*Cited in [Priority Sites](priority-sites.md) for typical survey-budget
heuristics.*

## Worked example reference

**Lee, S.-J., et al. (2025).** Breeding habitat prediction and
nest-site characteristics of the fairy pitta (*Pitta nympha*) in
Geoje-si, South Korea — Insights from a species distribution model.
*Global Ecology and Conservation*, 64, e03939.
*The published Java-MaxEnt analysis reproduced in the
[Pitta nympha worked example](examples/pitta-nympha.md).*

## How QMaxent cites itself

If you publish a study using QMaxent, please cite:

> Yu, B.-H. (2026). QMaxent: Maxent species distribution modeling in
> QGIS. *Software*. https://github.com/osgeokr/qmaxent

The exact CITATION.cff is the canonical source on the repository.
