# Literature novelty audit: high-risk analogues to joint LG-GWR

Status: **active literature audit; no final novelty claim accepted.**

This note records methods that are mathematically or conceptually close to joint LG-GWR and therefore constrain any future novelty claim.

## 1. User-facing narrative constraint

The phrase "supervised learning of local regression geometry" may describe an internal optimization mechanism, but should **not** be treated as the paper's first-level scientific framing at this stage. It sounds strongly machine-learning oriented and remains under conceptual review.

The first-level paper narrative should continue to center on geographic locality, spatial nonstationarity, geographic context, and effective/process-conditioned geographic geometry.

## 2. Contextualized GWR (Harris, Dong & Zhang, 2013)

Paper: *Using Contextualized Geographically Weighted Regression to Model the Spatial Heterogeneity of Land Prices in Beijing, China*, Transactions in GIS 17(6):901-919.

Key point: geographically close observations may be contextually/socially distant. Context variables are used to modify the geographic weighting matrix so proximity reflects both geographic distance and an attribute/context space.

Implication: joint LG-GWR cannot claim that it is the first GWR approach to incorporate geographic context into locality.

## 3. Spatial-attribute weighted GWR (Shi, Zhang & Liu, 2006)

Paper: *A new spatial-attribute weighting function for geographically weighted regression*, Canadian Journal of Forest Research 36(4):996-1005.

Key point: combines geographic space and attribute space in GWR weighting.

Implication: "space + attributes in GWR weights" is a long-established idea and cannot be the main novelty claim.

## 4. Spatial/temporal/context generalized distances

GTWR and contextual GTWR extensions use weighted combinations of spatial, temporal and contextual distances. One representative contextual GTWR formulation combines squared contextual, spatial and temporal distances with separate scale parameters.

Implication: constructing a higher-dimensional generalized proximity from multiple dimensions is not by itself novel.

## 5. Full bandwidth-matrix SVC (Hu et al., 2021)

Paper: *Selection of the Bandwidth Matrix in Spatial Varying Coefficient Models to Detect Anisotropic Regression Relationships*, Mathematics 9(18):2343.

Key point: a full two-dimensional bandwidth matrix is used in an SVC/local-linear GWR framework to represent anisotropic smoothing.

Implication: matrix-valued Mahalanobis-type geographic distance and learned anisotropy in the 2D coordinate domain are not new by themselves.

Important distinction from current joint LG-GWR: Hu et al.'s matrix acts on geographic coordinates `(u,v)` only, whereas joint LG-GWR acts on concatenated geographic position and process-relevant context.

## 6. SANNWR (Ni et al., 2022) — major high-risk analogue

Paper: *Spatial and Attribute Neural Network Weighted Regression for the Accurate Estimation of Spatial Non-Stationarity*, ISPRS International Journal of Geo-Information 11(12):620.

This is a much closer analogue than ordinary SGWR.

SANNWR explicitly:

- computes spatial distance and attribute distance;
- feeds them into a neural network (SAPNN/SAPDNN);
- learns a nonlinear "spatial-attribute unified distance metric";
- then uses this learned unified proximity in neural-network weighted regression.

Therefore joint LG-GWR **must not claim**:

- first learned spatial-attribute distance in GWR;
- first unified spatial-attribute proximity;
- first nonlinear/data-driven fusion of spatial and attribute proximity;
- first method in which model training changes GWR proximity.

### Structural distinction from joint LG-GWR

SANNWR learns an unrestricted neural fusion of precomputed spatial distance and attribute distance:

`d_SA = f_NN(d_S, d_A)`.

Current joint LG-GWR instead defines a single explicit PSD geometry on the concatenated coordinate-context differences:

`d_ij^2 = delta_u^T M delta_u`, where `u=[s,c]` and `M >= 0`.

Partitioning

`M = [[M_ss, M_sc], [M_cs, M_cc]]`

gives an explicit cross block `M_sc`, allowing spatial displacement and contextual difference to interact inside the quadratic geometry rather than only fusing two already-computed scalar distances.

This difference is potentially important because the LG-GWR object is an explicit geometric metric with identifiable invariants, whereas SANNWR is a flexible black-box proximity function. Whether this difference is scientifically sufficient must still be demonstrated rather than assumed.

## 7. GSAWR / CNN spatial-attribute GWR (Xu et al., published online 2025; IJGIS 2026 issue) — major high-risk analogue

Paper: *Geographically weighted regression with convolutional neural networks to integrate attribute similarity and spatial proximity*, International Journal of Geographical Information Science, 40(7), 2190-2215.

Key components:

- attribute-fusion CNN for multidimensional attribute similarities;
- spatial-attribute joint proximity neural network;
- learned joint proximity combining spatial proximity and attribute similarity;
- comparisons against GWR, SGWR, GNNWR, SANNWR and other neural variants.

Implication: even multi-attribute learned spatial-attribute joint proximity is now established in the GWR literature.

Therefore "joint proximity" or "learned unified proximity" alone is **not** a safe novelty claim for LG-GWR.

## 8. CEGWR 2026

CEGWR constructs high-dimensional contextual descriptors involving location/distance structure, direction and attributes, whitens them, and defines contextual distance in the resulting coordinate system.

Implication: non-separable/context-entangled geographic weighting has also entered recent GWR literature.

## 9. Spatial deformation and warped-space lineage

Relevant lineage includes:

- Eldridge & Jones (1991), *Warped Space: A Geography of Distance Decay*;
- Sampson & Guttorp (1992), spatial deformation for nonstationary covariance;
- Schmidt & O'Hagan (2003), Bayesian spatial deformation;
- Schmidt, Guttorp & O'Hagan (2011), covariates in spatial covariance/deformation.

Eldridge & Jones are especially useful geographically: equivalent physical distances can have spatially uneven interaction effects, and distance decay can be contextual rather than universal.

Implication: "warped geographic space" is an established geographic/spatial-statistical idea, not an LG-GWR invention.

## 10. Current narrowed novelty candidate for joint LG-GWR

After this audit, none of the following is individually safe as the main novelty claim:

- adding attributes/context to GWR;
- spatial + attribute weighting;
- contextualized locality;
- non-Euclidean distance;
- Mahalanobis distance;
- anisotropy;
- bandwidth matrix;
- data-driven GWR weights;
- learned spatial-attribute unified distance;
- neural/nonlinear joint proximity;
- latent/deformed geographic space.

The currently surviving structural candidate is narrower:

> **an explicit, low-dimensional, interpretable PSD geometry defined directly on the joint geographic-context coordinate system, with non-separable geography-context cross terms, used to characterize locality of spatially varying regression relationships.**

Mathematically:

`u=[s,c]`

`d_ij^2 = (u_i-u_j)^T M (u_i-u_j)`

with

`M = [[M_ss, M_sc],[M_cs,M_cc]]`.

The potential distinguishing object is not merely the existence of a joint proximity, but the explicit geometric structure and especially the interpretable cross-coupling `M_sc`.

This candidate still requires additional checks:

1. search for high-dimensional SVC/local-polynomial bandwidth matrices operating on `[space, context]` rather than space alone;
2. search for explicit PSD/Mahalanobis metric learning in GWR/SVC;
3. search for regression-specific spatial deformation of coefficient surfaces;
4. establish whether explicit metric structure yields scientific advantages over neural joint-proximity methods (stability, invariance, interpretability, parsimonious parameterization, null behavior, geographic mapping);
5. avoid claiming priority until these checks are complete.

## 11. Scientific framing remains open

Even if the algorithm internally optimizes `M` using a regression objective, the paper need not foreground "supervised metric learning". The scientific framing under consideration remains:

- what constitutes locality for spatially nonstationary relationships;
- whether equal physical distances can imply unequal relationship proximity under different geographic contexts;
- whether spatially varying relationships are better represented in an effective geographic geometry than in fixed Euclidean space.

No final terminology has yet been accepted.
