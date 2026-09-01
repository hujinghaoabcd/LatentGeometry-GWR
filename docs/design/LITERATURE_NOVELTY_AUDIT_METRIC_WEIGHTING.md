# Literature novelty audit: metric, weighting, and anisotropy around LG-GWR

Status: literature audit only; no algorithm decision.

## Scope

This note records prior work most relevant to claims about learned distance, full Mahalanobis-style geometry, spatial-attribute weighting, and data-adaptive GWR weights. It is intended to prevent overclaiming novelty for joint LG-GWR.

## Key prior work

### Shi, Zhang & Liu (2006) — spatial-attribute weighted GWR

**A new spatial-attribute weighting function for geographically weighted regression**. Canadian Journal of Forest Research, 36(4), 996–1005. DOI: 10.1139/X05-295.

This paper already combines geographic space and attribute space in a GWR weighting function. Therefore LG-GWR must not claim that combining geographic proximity and attribute similarity is new.

### Hu et al. (2021) — full bandwidth matrix for spatial varying coefficient models

**Selection of the Bandwidth Matrix in Spatial Varying Coefficient Models to Detect Anisotropic Regression Relationships**. Mathematics, 9(18), 2343. DOI: 10.3390/math9182343.

The paper introduces an unconstrained 2x2 bandwidth matrix H for geographic coordinates (u,v), giving a Mahalanobis-form kernel and directional smoothing. The matrix is selected using plug-in/adaptive bandwidth procedures rather than a regression-task-learned high-dimensional geographic-context metric.

Important consequence: a full matrix-valued spatial bandwidth / anisotropic Mahalanobis geometry is not itself novel. Joint LG-GWR must distinguish itself through the enlarged space, cross-coupling with process-relevant context, and the way the geometry is estimated.

### Páez (2004) and Yan et al. (2024) — anisotropic GWR

Páez develops anisotropic variance functions in GWR. Yan, Wu & Duan (2024), **Modeling Spatial Anisotropic Relationships Using Gradient-Based Geographically Weighted Regression**, Annals of the American Association of Geographers, introduces locally varying gradient information to model anisotropic relationships.

Therefore claims such as 'first GWR to model anisotropy' or 'first to move beyond Euclidean isotropy' are invalid.

### Lu et al. / PSDM-GWR line

Minkowski and parameter-specific distance-metric GWR papers show that GWR distance metrics can be calibrated and can differ across regression relationships. This literature occupies the 'distance metric is a model component rather than fixed Euclidean geometry' space.

### Du et al. (2020) — GNNWR

**Geographically neural network weighted regression for the accurate estimation of spatial non-stationarity**, IJGIS 34(7), 1353–1377.

GNNWR already learns a nonstationary weight matrix using a neural network. GTNNWR extends this to space-time proximity. Therefore 'data-driven/learned GWR weights' alone cannot be the core novelty claim of LG-GWR.

### Zhang et al. (2025) — supervised spatial metric learning

**Supervised spatial metric learning with applications to spatial clustering and spatial model prediction**, Journal of Geographical Systems 27(4), 523–553.

A response-informed spatial similarity matrix is learned and then used with models including GWR. Thus 'supervised spatial similarity learning used with GWR' is also not sufficient as a novelty claim.

### SGWR family and earlier CWR/GAWR

Shi et al. (2006), CWR (2023), SGWR (2024), SGWR-GD (2025), later efficient SGWR implementations, and SGNNWR (2026) all combine or learn geographic and attribute similarity in some form.

The geography-preserving/separable LG-GWR path therefore faces direct novelty competition from this family.

## Current novelty boundary for joint LG-GWR

The audit has not found, in this search round, a prior GWR/SVC method matching all of the following simultaneously:

1. construct a single joint vector containing geographic position and process-relevant contextual descriptors;
2. estimate one coupled PSD metric over that joint space;
3. allow nonzero geographic-context cross-block terms M_sc, so spatial displacement and contextual difference interact inside the distance itself rather than through post-hoc fusion of two similarities;
4. use that geometry directly to define locality for spatially varying regression relationships;
5. retain an interpretable local linear-regression output rather than replacing the local model with a general neural predictor.

This is a provisional novelty boundary, not a first-in-literature claim. Further searching is required, especially for high-dimensional bandwidth-matrix SVC, contextual Mahalanobis GWR, relationship-dependent spatial warping, and regression-specific spatial deformation.

## Terminology caution

Do not currently lead the paper with phrases such as 'supervised learning of local regression geometry'. Although technically descriptive, that framing makes the method read as machine learning. For now, keep the scientific layer centered on spatial nonstationarity, effective geographic locality, and process-related geographic geometry. The optimization mechanism can be described later in the methods section.

## Claims currently unsafe

Do not claim:

- first GWR combining geographic and attribute space;
- first non-Euclidean or anisotropic GWR;
- first matrix-valued / Mahalanobis GWR geometry;
- first data-driven GWR weights;
- first supervised spatial similarity used with GWR;
- first use of geographic context to modify GWR locality.

## Candidate claim still worth testing

A potentially defensible joint-LG-GWR contribution is a **single non-separable geographic-context geometry for spatially varying relationships**, especially the explicit geographic-context cross-coupling represented by the off-diagonal block M_sc, provided later literature search and experiments confirm that this structure is not already present in a closely equivalent GWR/SVC formulation.
