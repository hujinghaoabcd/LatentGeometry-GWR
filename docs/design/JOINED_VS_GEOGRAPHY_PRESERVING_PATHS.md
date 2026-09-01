# LG-GWR candidate theory paths: joint geometry vs geography-preserving locality

Status: **open design question — both paths are retained; no final model choice has been made.**

This note records two competing but potentially complementary directions for the LG-GWR paper. Neither should be removed merely because the other currently appears more attractive. The final choice must follow literature review, mathematical audit, targeted simulation, and geographic case evidence.

## 1. Shared scientific problem

Both paths start from the same geographic question:

> Does geographic proximity alone adequately define locality for spatially nonstationary relationships?

The paper should remain centered on geographic locality, spatial nonstationarity, and process-relevant geographic context rather than on metric learning as an end in itself.

Preferred geographic framing:

- spatial proximity defines one aspect of locality;
- geographic context may alter which nearby or non-nearby observations contribute meaningfully to a local relationship;
- the target is to characterize **effective / process-conditioned geographic locality**;
- machine-learning components are computational means, not the scientific object.

Terminology note: avoid Chinese expressions such as “空间借样” or “信息借用”. Prefer “局地信息整合”, “邻域贡献”, “局部样本权重”, “有效邻域构成”, or “局地权重分配” depending on context.

## 2. Path A — Joint LG-GWR as learned geographic geometry

### 2.1 Core formulation

Let

`u_i = [s_i, c_i]`

where `s_i` is geographic position and `c_i` is process-relevant geographic context. Joint LG-GWR learns

`z_i = A u_i`

and defines locality through

`d_ij^LG = ||A(u_i-u_j)||`.

Equivalently, with `M=A^T A`,

`(d_ij^LG)^2 = delta_u^T M delta_u`.

Partitioning `M` by geographic and contextual variables gives

`M = [[M_ss, M_sc], [M_cs, M_cc]]`

so that

`d^2 = delta_s^T M_ss delta_s + 2 delta_s^T M_sc delta_c + delta_c^T M_cc delta_c`.

The cross block `M_sc` is the key structural distinction: space and context need not contribute as two separable similarity channels.

### 2.2 Main innovation claim to preserve

Joint LG-GWR should not be described merely as “GWR + attribute similarity”. Its stronger candidate interpretation is:

> **process-conditioned deformation / reconstruction of geographic geometry for local regression.**

The geographic meaning of a physical separation is allowed to depend on contextual differences, and contextual differences are allowed to have different effects depending on spatial relations. This is a non-separable space–context geometry.

Candidate scientific formulation:

`G_process = T(G_physical, C_process)`

where the model estimates an effective geometry associated with a spatially nonstationary relationship.

### 2.3 Why this can be more innovative than SGWR

SGWR-type methods generally retain a geographic proximity channel and an attribute-similarity channel and combine them. Joint LG-GWR instead learns one coupled geometry. The potential distinction is therefore:

- SGWR: `geographic similarity + attribute similarity`;
- joint LG-GWR: `non-separable geographic-context geometry`;
- SGWR combines two predefined/provided notions of proximity;
- joint LG-GWR estimates how geographic and contextual dimensions jointly define locality from the regression objective.

The strongest algorithmic novelty is therefore not “data-driven weights” alone, but the learned cross-coupling between geographic position and process-relevant context.

### 2.4 Main risks

This path also has the larger theoretical risk:

1. physical geography can in principle be strongly deformed or nearly suppressed;
2. if interpreted carelessly, the method becomes generic supervised metric learning;
3. latent axes are not uniquely identifiable;
4. metric/bandwidth scale identification must be handled explicitly;
5. recent contextual-geometry methods such as CEGWR may overlap in high-level motivation;
6. geographic validity must be demonstrated, not assumed from improved prediction.

Thus joint LG-GWR currently has **higher algorithmic novelty but higher geographic/theoretical burden of proof**.

## 3. Path B — Geography-preserving / separable process-conditioned locality

### 3.1 Core formulation

The geography-preserving family keeps physical geographic distance explicit and adds a learned contextual component, schematically:

`w_ij = K_g(d_geo,ij / h_g) * K_c(d_context,ij / h_c)`

or another constrained modulation in which geographic support cannot disappear.

Current separable LG-GWR is one baseline realization:

`d_context,ij = ||B(c_i-c_j)||`.

### 3.2 Main geographic strength

This formulation has a clearer geographical interpretation:

> physical spatial proximity remains the geographic support of local regression, while process-relevant geographic context modifies the composition and relative contribution of observations within that support.

This path aligns naturally with the GWR tradition, Tobler-style proximity, geographic similarity/configuration arguments, and spatial-process interpretation.

### 3.3 Main innovation challenge

Its main weakness is novelty overlap. SGWR, SGWR-GD and related methods already combine geographic proximity with attribute/context similarity. A simple product of geographic and contextual kernels is therefore not sufficient as the final contribution.

If this route becomes the main model, its novelty must come from at least one stronger element, for example:

- process-supervised learning of the contextual metric rather than a fixed similarity formula;
- explicit distinction between explanatory variables `X` and geographic-context descriptors `C`;
- geographic constraints with learned contextual modulation;
- parameter-/process-specific contextual locality rather than one generic attribute similarity;
- theory connecting contextual locality to spatially varying relationships rather than only predictive improvement.

Thus geography-preserving LG-GWR currently has **stronger geographic defensibility but potentially weaker raw algorithmic novelty**.

## 4. The two paths must not be collapsed prematurely

At this stage the project explicitly retains both hypotheses:

### Hypothesis A — joint geometry may be the main innovation

The strongest method may be joint LG-GWR because it learns a genuinely non-separable geographic-context geometry and can be clearly distinguished from SGWR-style weight fusion.

### Hypothesis B — geography-preserving locality may be the scientifically safer formulation

The strongest paper may instead require physical geography to remain an explicit anchor, because this yields a clearer geographic interpretation and reduces the risk that LG-GWR becomes generic metric learning.

These are not contradictory research records. They define the main model-selection question to be resolved empirically and theoretically.

## 5. Required comparison framework

Do not select the main formulation from benchmark accuracy alone. The comparison should include:

1. **geographic faithfulness** — does the learned locality retain plausible spatial structure?
2. **process sensitivity** — does locality change when the underlying geographic process changes?
3. **null behavior** — when context is irrelevant, does the model return toward ordinary GWR?
4. **boundary behavior** — can the method represent sharp geographic/contextual transitions without arbitrary artifacts?
5. **analogue behavior** — can similar process contexts produce related locality structures where scientifically justified?
6. **identifiability/stability** — are learned geometric objects stable under scaling, initialization, and equivalent parameterizations?
7. **interpretability** — can the learned locality be explained through maps and geographic context rather than only loss values?
8. **novelty against literature** — especially SGWR, SGWR-GD, EDSGWR, CEGWR, supervised spatial metric learning, non-Euclidean GWR, PSDM-GWR, and MGWR.
9. **predictive/generalization performance** — important, but not the sole decision criterion.

## 6. Experimental role of each model if Path A remains primary

A useful provisional hierarchy is:

- ordinary GWR — fixed geographic-locality baseline;
- SGWR / related similarity-GWR — separable geographic + attribute-similarity literature baseline;
- current separable LG-GWR — learned contextual metric but separable geography/context structure;
- joint LG-GWR — learned non-separable geographic-context geometry;
- optional geography-constrained joint variant — tests whether joint interaction can be retained while preventing loss of geographic anchoring.

This design lets the paper test whether the `M_sc` cross-coupling adds genuine geographic-process information beyond separate similarity channels.

## 7. Paper narrative to preserve regardless of final algorithm

The first-level narrative should remain geographic:

`spatial heterogeneity -> spatially varying relationships -> locality assumption -> limits of geographic distance alone -> process-relevant geographic context -> effective/process-conditioned geographic locality -> LG-GWR`

Do not lead with:

`latent embedding -> Adam -> metric learning -> prediction improvement`.

The algorithmic machinery belongs in the method section after the geographic question has been established.

## 8. Current decision status

**NO FINAL DECISION. BOTH PATHS RETAINED.**

Current provisional assessment:

- joint LG-GWR: stronger algorithmic distinctiveness, especially relative to SGWR, due to non-separable space–context geometry and cross terms;
- geography-preserving/separable LG-GWR: stronger immediate geographic interpretability and theoretical safety, but more direct competition with recent similarity-GWR literature.

The next research stage should be designed to discriminate between these two possibilities rather than assume either one is correct.
