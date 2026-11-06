# Proposal: Threshold Exceedance as Part of the Analysis Pipeline

*In plain words: the threshold bars currently show how often infants are highly similar in each brain region, but they are only drawn, never tested. This proposal turns them into a defined quantity of the analysis, with confidence intervals, significance tests and tables, and extends the same idea to individuality.*

Notation follows `ANALYSIS_PIPELINE.md` (N = 19 infants, 171 infant pairs per session, regions and region pairs as in its Section 3.1).

---

## 1. Current status

The threshold bars are produced only in `09_generate_figures.py`. For each session, region (nodes) or region pair (edges), the script pools the element similarities J of all 171 infant pairs and plots the percentage above 0.30, 0.50, 0.75 and 0.90, stacked by hemispheric category. They are not in the analysis plan, they are computed at plotting time, and they carry no confidence intervals, tests or tables. The manuscript equation for this quantity (former Eq. 5) is on hold.

## 2. What it adds to the questions we are answering

1. **How widespread high similarity is, not only how high it is on average.** U_s(g) is the average similarity in region g. Two regions with the same average can differ: in one, every infant pair is moderately similar; in the other, some pairs are near-identical and others dissimilar. Threshold exceedance distinguishes these cases.
2. **It contains the average.** The area under the exceedance curve (Section 3.4) equals the mean similarity of the region. Exceedance therefore refines U_s(g) rather than competing with it.
3. **It extends to individuality (optional).** The same count, applied to an infant's own cross-session comparisons and to comparisons with other infants, gives a threshold version of I_diff (Section 3.5).

---

## 3. Definitions

| Symbol | Meaning |
| :---: | :--- |
| θ | Similarity threshold: 0.30, 0.50, 0.75 or 0.90, fixed in advance (the legacy values). Unrelated to the theta band. |
| g | A region (5 for node features) or region pair (15 for edge features). |
| n_g | Number of elements in g. |
| h | Hemispheric category: L, R, M (nodes); L-L, R-R, L-R, L-M, R-M, M-M (edges). |
| [·] | 1 if the condition inside holds, 0 otherwise. |
| n^def | Number of defined values (not 0/0) used as a denominator. |

### 3.1 Exceedance per region (Comparison A, universality)

Pool every infant pair and every element of g (171 × n_g values) and count how many exceed θ:

$$
\Phi_{g,s}(\theta) = \frac{100}{n^{\text{def}}_{g,s}} \sum_{a<b} \sum_{e \in g} \big[J_{(a,s),(b,s)}(e) > \theta\big]
$$

The percentage of all infant-pair comparisons in g, at session s, whose similarity is above θ (strict >).

*Example:* F-F has 21 edges, so 171 × 21 = 3,591 values. If 2,700 exceed 0.75, Φ = 75.2%.

### 3.2 Split by hemispheric category

Two versions, because they answer different questions.

**Contribution** (the stacked segments of the bars): same denominator as 3.1, so the segments of a bar add up to Φ_{g,s}(θ).

$$
\Phi_{g,h,s}(\theta) = \frac{100}{n^{\text{def}}_{g,s}} \sum_{a<b} \sum_{e \in g \cap h} \big[J_{(a,s),(b,s)}(e) > \theta\big]
$$

A segment can never exceed category h's share of the region. L-R is 9 of the 21 F-F edges, so its segment is at most 42.9%; it mixes how often the category exceeds θ with how large the category is.

**Rate within the category**: divided by the defined values of that category only, so it is the percentage of the category's own comparisons above θ.

$$
\Phi_{g \mid h,s}(\theta) = \frac{100}{n^{\text{def}}_{g \cap h,s}} \sum_{a<b} \sum_{e \in g \cap h} \big[J_{(a,s),(b,s)}(e) > \theta\big]
$$

Use the rate, not the contribution, to compare hemispheric categories with each other.

### 3.3 Change between sessions

$$
\Delta\Phi_g(\theta) = \Phi_{g,2}(\theta) - \Phi_{g,1}(\theta)
$$

The exceedance counterpart of ΔU: positive when high similarity is more widespread at the second visit.

### 3.4 Exceedance curve

Compute Φ_{g,s}(θ) for θ = 0, 0.01, …, 0.99. Every J lies in [0, 1], and for such a value J = ∫₀¹ [J > θ] dθ. Averaging over the values of g gives

$$
\int_0^1 \frac{\Phi_{g,s}(\theta)}{100} \, d\theta = \text{mean of the defined } J \text{ values in } g
$$

With no 0/0 values this mean is exactly U_s(g). The four fixed thresholds are points on this curve, so no conclusion depends on choosing among them.

### 3.5 Individuality extension (optional)

Apply the same count to the element matrices M_e of Phase 4 (rows ses-1, columns ses-2): the diagonal holds each infant compared with itself, the off-diagonal holds different infants.

$$
\Phi^{\text{self}}_g(\theta) = \frac{100}{n^{\text{def,self}}_g} \sum_{a=1}^{19} \sum_{e \in g} \big[M_e[a,a] > \theta\big],
\qquad
\Phi^{\text{others}}_g(\theta) = \frac{100}{n^{\text{def,others}}_g} \sum_{a \ne b} \sum_{e \in g} \big[M_e[a,b] > \theta\big]
$$

$$
\Delta\Phi^{\text{ident}}_g(\theta) = \Phi^{\text{self}}_g(\theta) - \Phi^{\text{others}}_g(\theta)
$$

n^def,self is at most 19 × n_g and n^def,others at most 342 × n_g. In words: how much more often an infant's element is highly similar to its own earlier value than to another infant's. As with I_diff, the subtraction removes development shared by all infants.

### 3.6 The 0/0 rule and the denominator

Under the current main rule, 0/0 comparisons are NaN and are left out of both the count and n^def, so Φ describes agreement only where at least one infant has a non-zero value. Under the legacy rule (0/0 → 0, Step 8 check 1), n^def = 171 × n_g and the legacy notebook values are reproduced. For features without zeros the two rules give identical values. For the binary edge features, where about half of all comparisons are 0/0, they differ substantially.

---

## 4. Statistics

The 171 pairs are not independent, so every inference resamples or permutes infants, using the same resamples, shuffles and seed as the rest of the pipeline.

| Quantity | Uncertainty (95% CI) | Test | Multiple comparisons |
| :--- | :--- | :--- | :--- |
| Φ_{g,s}(θ), Φ_{g\|h,s}(θ) | Infant pair bootstrap: in each resample, the count above θ and n^def are both taken over the drawn pairs with d_p ≠ d_q (weights c_a c_b), and Φ is their ratio | Element-shuffle null of Phase 2 (same shuffles): is high similarity in g more common than when infants' elements are paired at random? One-sided p | FDR across regions × thresholds of each feature, per session (20 tests for a node feature, 60 for an edge feature) |
| ΔΦ_g(θ) | Paired pair bootstrap: same drawn pairs for both sessions | Two-sided bootstrap p | As above |
| Φ^self_g(θ), Φ^others_g(θ), ΔΦ^ident_g(θ) | Infant bootstrap of M_e: self over drawn infants, others over off-diagonal positions with d_p ≠ d_q | Column permutation of Phase 4 (same shuffles as I_diff(e)): is ΔΦ^ident > 0? One-sided p | As above |

The element-shuffle null can reuse the fast route of Phase 8: under a shuffle π, the epoch-averaged similarity at element e is G[e, π(e)], so the shuffled Φ is a count over these values.

---

## 5. Where it plugs into the pipeline

| Pipeline location | Change |
| :--- | :--- |
| Phase 2, `02_compute_universality.py` | New item 6: Φ_{g,s}(θ), both hemispheric splits, ΔΦ_g(θ), the curve, CIs, null test, FDR. Saved in `universality_results.json` under `exceedance`. |
| Phase 4, `04_compute_identifiability.py` | New item 6 (optional): Φ^self, Φ^others, ΔΦ^ident with CIs, permutation test, FDR. Saved in `identifiability_results.json` under `exceedance`. |
| Phase 7 | Two rows in the implementation table, as in Section 4. |
| Phase 8, check 1 (`07_robustness_checks.py`) | Recompute Φ under 0/0 → 0. |
| `08_export_tables.py` | Tables T2c and T2d (Section 6). |
| `09_generate_figures.py` | Threshold bars read Φ from `universality_results.json` instead of computing it; new exceedance-curve figure. |
| Manuscript Methods | Reinsert the equation for Φ (former Eq. 5) with the defined-values denominator. |

---

## 6. Outputs

**Table T2c (universality).** One row per feature, session, region and threshold: n_g, n^def, Φ_{g,s}(θ) as median [95% CI] with p and q; for each hemispheric category, the contribution and the rate [95% CI]; ΔΦ_g(θ) [95% CI] with p and q.

**Table T2d (individuality, optional).** One row per feature, region and threshold: Φ^self, Φ^others and ΔΦ^ident, each as median [95% CI], with p and q for ΔΦ^ident.

**Threshold bars** (existing figure, unchanged in appearance): values now read from the results, and the panel title states the number of 0/0 values left out.

**Exceedance curves** (new figure): for each region or region pair, Φ_{g,s}(θ) against θ from 0 to 1, ses-1 and ses-2 overlaid with 95% bootstrap bands, the four fixed thresholds marked; a second panel with Φ^self and Φ^others. Key features in the main figure, all 26 in a supplement.

---

## 7. Decisions needed

1. **0/0 rule.** Φ's denominator follows the main 0/0 rule, which is still open. For binary features the choice changes the percentages substantially (Section 3.6).
2. **Individuality extension.** Include Φ^self, Φ^others and ΔΦ^ident (Section 3.5), or keep exceedance to universality only.
3. **Thresholds.** Keep 0.30, 0.50, 0.75 and 0.90 as the reported values, with the curve as the threshold-free summary.
4. **Hemispheric reporting.** Report the within-category rate alongside the stacked contributions, so categories of different sizes can be compared.
