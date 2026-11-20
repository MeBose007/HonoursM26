"""
0/0 rule comparison: every result that depends on the 0/0 rule, computed under both rules.

The weighted Jaccard similarity of two values is min / max. When both values are 0 this is 0/0:
  - 0/0 = NaN (main analysis): the comparison is undefined and left out of every mean.
  - 0/0 = 0 (legacy notebooks, Phase 7 check 1): the comparison counts as similarity 0.
The rule is applied to the element-level J and to the recording-level S, exactly as in Phase 7 check 1.

A feature without zero values can never produce 0/0, so its results are identical under both rules; this is
checked below. For the features with zeros, this script recomputes, under each rule and with the same seed,
permutations and bootstrap resamples (B = config.N_PERMUTATIONS):
  - Comparison A (Phase 2): U_s, Delta U, U_s(e) with the element-shuffle test, Delta U(e), regional U_s(g)
  - Comparison B (Phase 3, primary split): W_s, O_s, W_s - O_s and W_s(e) - O_s(e)
  - Comparisons C and D (Phase 4): I_self, I_others, I_diff, 1-NN accuracy and I_diff(e) with their tests
  - The Step 5a classes (Phase 5) and the threshold exceedance shown in the threshold bars (Phase 9)
The recording-level values and the identifiability tests are computed for all 26 features.

The element-shuffle test uses the Phase 7 route (U_s(e) of a shuffle read from the epoch-averaged cross-element
similarities G[e, pi(e)]), which has the same null distribution as Phase 2 but draws its permutations differently,
so its p-values agree with Phase 2 up to Monte Carlo error. Every other step follows Phases 2-5 exactly; with
0/0 = NaN it reproduces the saved pipeline values, which is checked and written to validation_vs_pipeline.csv.

Outputs: zero_rule_comparison/results/*.csv and zero_rule_comparison/figures/*.
Run from the project root: .venv/bin/python zero_rule_comparison/compare_zero_rules.py [--permutations N] [--workers N]
"""
import argparse
import importlib
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis_pipeline.config import (
    FEATURE_DEFS,
    N_EPOCHS,
    N_PERMUTATIONS,
    OUTPUT_TENSORS_DIR,
    PERM_TIE_TOL,
    SEED,
    SESSIONS,
    SUBJECT_IDS,
)
from analysis_pipeline.utils import (
    bootstrap_ci,
    bootstrap_counts,
    bootstrap_draws,
    bootstrap_p,
    identification_accuracy,
    recording_S,
    weighted_nanmean,
)

robust = importlib.import_module("analysis_pipeline.07_robustness_checks")  # jaccard_ratio, cross_J, nan_mean, element_shuffle_p
univ = importlib.import_module("analysis_pipeline.02_compute_universality")  # region_groups
figs = importlib.import_module("analysis_pipeline.09_generate_figures")  # element_info, hemi_category, feature_title

HERE = Path(__file__).resolve().parent
RESULTS_DIR, FIG_DIR = HERE / "results", HERE / "figures"
RULES = ("nan", "zero")
RULE_LABEL = {"nan": "0/0 = NaN", "zero": "0/0 = 0"}
RULE_COLOR = {"nan": "#2b6cb0", "zero": "#dd8452"}
THRESHOLDS = (0.30, 0.50, 0.75, 0.90)
CLASSES = ["Universal", "Individual", "Universal and Individual", "Neither"]
N = len(SUBJECT_IDS)
PAIRS = list(combinations(range(N), 2))
P_IDX, Q_IDX = np.array(PAIRS).T
K = np.arange(N_EPOCHS)  # epoch k with epoch k, as in the main analysis
FIRST, SECOND = np.arange(N_EPOCHS // 2), np.arange(N_EPOCHS // 2, N_EPOCHS)  # Phase 3 primary split, k with k + 6


# ----------------------------------------------------------------------------------------------
# Per feature and rule: everything that needs the feature tensor (run in worker processes)
# ----------------------------------------------------------------------------------------------

def zero_counts(X: np.ndarray) -> dict:
    """How often both compared values are 0 (0/0): single-epoch comparisons, pair-level J, recording-level S."""
    row = {"Zero values in X (%)": 100.0 * float(np.mean(X == 0))}
    for s, ses in enumerate(SESSIONS):
        mx = np.maximum(X[P_IDX, s], X[Q_IDX, s])  # (pairs, epochs, dim), epoch k with epoch k
        row[f"Epoch comparisons 0/0 {ses} (%)"] = 100.0 * float(np.mean(mx == 0))
        row[f"Pair J(e) undefined {ses}"] = int(np.all(mx == 0, axis=1).sum())
        row[f"S epoch pairs 0/0 {ses}"] = int(np.all(mx == 0, axis=2).sum())
    mx = np.maximum(X[:, None, 0], X[None, :, 1])  # (a at ses-1, b at ses-2, epochs, dim)
    row["Epoch comparisons 0/0 cross-session (%)"] = 100.0 * float(np.mean(mx == 0))
    row["M_e cells undefined"] = int(np.all(mx == 0, axis=2).sum())
    row["S epoch pairs 0/0 cross-session"] = int(np.all(mx == 0, axis=3).sum())
    return row


def identifiability_test(M: np.ndarray, M_elem: np.ndarray, n_perm: int) -> dict:
    """
    Phase 4 items 2-5: I_self, I_others, I_diff, 1-NN accuracy and I_diff(e), each with its one-sided p from column
    permutations of M and of every M_e together, drawn from the recorded seed in the same order as Phase 4.
    """
    n = M.shape[0]
    rows = np.arange(n)
    diag = np.eye(n, dtype=bool)
    out = {"I_self": float(np.nanmean(M[diag])), "I_others": float(np.nanmean(M[~diag]))}
    out["I_diff"] = out["I_self"] - out["I_others"]
    out["acc_forward"], out["acc_backward"] = identification_accuracy(M)
    out["acc_mean"] = (out["acc_forward"] + out["acc_backward"]) / 2.0
    I_diff_elem = np.nanmean(M_elem[diag, :], axis=0) - np.nanmean(M_elem[~diag, :], axis=0)

    M_valid = ~np.isnan(M)
    M_filled = np.where(M_valid, M, 0.0)
    total_M_sum, total_M_count = M_filled.sum(), M_valid.sum()
    total_elem_sum = np.nansum(M_elem, axis=(0, 1))
    total_elem_count = np.sum(~np.isnan(M_elem), axis=(0, 1))
    rng = np.random.default_rng(SEED)
    ge = dict.fromkeys(("I_diff", "acc_forward", "acc_backward", "acc_mean"), 0)
    ge_elem = np.zeros(M_elem.shape[2], dtype=np.int64)
    for _ in range(n_perm):
        col_perm = rng.permutation(n)
        s_sum, s_count = M_filled[rows, col_perm].sum(), M_valid[rows, col_perm].sum()
        ge["I_diff"] += s_sum / s_count - (total_M_sum - s_sum) / (total_M_count - s_count) >= out["I_diff"] - PERM_TIE_TOL
        f, b = identification_accuracy(M[:, col_perm])
        ge["acc_forward"] += f >= out["acc_forward"] - PERM_TIE_TOL
        ge["acc_backward"] += b >= out["acc_backward"] - PERM_TIE_TOL
        ge["acc_mean"] += (f + b) / 2.0 >= out["acc_mean"] - PERM_TIE_TOL
        diag_vals = M_elem[rows, col_perm, :]
        s_elem = np.nanmean(diag_vals, axis=0)
        diag_sum, diag_count = np.nansum(diag_vals, axis=0), np.sum(~np.isnan(diag_vals), axis=0)
        off_count = total_elem_count - diag_count
        o_elem = np.where(off_count > 0, (total_elem_sum - diag_sum) / off_count, np.nan)
        ge_elem += (s_elem - o_elem >= I_diff_elem - PERM_TIE_TOL).astype(np.int64)
    for key, count in ge.items():
        out[f"p_{key}"] = (1.0 + count) / (1.0 + n_perm)
    p_elem = (1.0 + ge_elem) / (1.0 + n_perm)
    p_elem[np.isnan(I_diff_elem)] = 1.0  # 0/0 in every comparison: no observed I_diff(e)
    out["I_diff_elem"], out["p_I_diff_elem"] = I_diff_elem, p_elem
    return out


def half_J(X_s: np.ndarray, a: int, b: int, zero_rule: bool) -> np.ndarray:
    """J(e) between infant a's first half and infant b's second half of one session (epoch k with k + 6, Phase 3)."""
    return robust.nan_mean(robust.jaccard_ratio(X_s[a, FIRST], X_s[b, SECOND], zero_rule), axis=0)


def rule_results(task):
    """All rule-dependent arrays of one feature under one rule. with_shuffle: run the element-shuffle test of U_s(e)."""
    fid, rule, n_perm, with_shuffle = task
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    zero_rule = rule == "zero"
    X = np.load(OUTPUT_TENSORS_DIR / f"X_{fid}.npy")  # (19, 2, 12, dim)
    res = {}
    rng = np.random.default_rng(SEED)  # element-shuffle permutations, ses-1 then ses-2, as in Phase 7
    for s, ses in enumerate(SESSIONS):
        r = {}
        # Comparison A: J of every infant pair and element, and U_s(e) with its element-shuffle p
        G = np.stack([robust.cross_J(X[a, s], X[b, s], K, K, zero_rule) for a, b in PAIRS])
        r["pair_J"] = np.diagonal(G, axis1=1, axis2=2).copy()  # (171, dim)
        r["U_elem"] = robust.nan_mean(r["pair_J"], axis=0)
        r["p_U_elem"] = robust.element_shuffle_p(G, r["U_elem"], n_perm, rng) if with_shuffle else None
        del G
        r["pair_S"] = np.array([recording_S(X[a, s], X[b, s], zero_rule) for a, b in PAIRS])
        # Comparison B (Phase 3 primary split): an infant's own two halves (W), and its first half against every
        # other infant's second half (O, averaged over the 18 others)
        X_s = X[:, s]
        others = [[b for b in range(N) if b != a] for a in range(N)]
        r["W_rec"] = np.array([recording_S(X_s[a, FIRST], X_s[a, SECOND], zero_rule) for a in range(N)])
        r["W_elem"] = np.array([half_J(X_s, a, a, zero_rule) for a in range(N)])
        r["O_rec"] = np.array([np.nanmean([recording_S(X_s[a, FIRST], X_s[b, SECOND], zero_rule) for b in others[a]])
                               for a in range(N)])
        r["O_elem"] = np.array([np.nanmean([half_J(X_s, a, b, zero_rule) for b in others[a]], axis=0) for a in range(N)])
        res[ses] = r
    # Comparisons C and D: cross-session S (matrix M) and element J (matrices M_e), tested as in Phase 4
    M = np.array([[recording_S(X[a, 0], X[b, 1], zero_rule) for b in range(N)] for a in range(N)])
    M_elem = np.array([[robust.nan_mean(robust.jaccard_ratio(X[a, 0], X[b, 1], zero_rule), axis=0) for b in range(N)]
                       for a in range(N)])
    res["M"] = M
    res.update(identifiability_test(M, M_elem, n_perm))
    return fid, rule, res


# ----------------------------------------------------------------------------------------------
# Bootstrap statistics (main process, same infant resamples as Phases 2 and 3)
# ----------------------------------------------------------------------------------------------

def summarise(fid: str, rule: str, res: dict, counts: np.ndarray, W_pairs: np.ndarray) -> dict:
    """Recording-level values with their tests, and the element-level and regional statistics, for one rule."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    groups = univ.region_groups(FEATURE_DEFS[fid]["type"])
    out = {"global": {}, "elem": {}, "region": {}, "exceed": {}}
    g, e = out["global"], out["elem"]
    boot_S, boot_J = {}, {}
    for s, ses in enumerate(SESSIONS, 1):
        r = res[f"ses-{s}"]
        # Comparison A (Phase 2)
        g[f"U_{s}"] = float(np.nanmean(r["pair_S"]))
        boot_S[s] = weighted_nanmean(W_pairs, r["pair_S"])
        boot_J[s] = weighted_nanmean(W_pairs, r["pair_J"])
        e[f"U_{s}(e)"] = r["U_elem"]
        if r["p_U_elem"] is not None:
            e[f"p U_{s}(e)"] = r["p_U_elem"]
            e[f"q U_{s}(e)"] = np.asarray(multipletests(r["p_U_elem"], method="fdr_bh")[1])
        e[f"Pairs with J(e) undefined, ses-{s}"] = np.isnan(r["pair_J"]).sum(axis=0)
        J_region = np.column_stack([np.nanmean(r["pair_J"][:, idxs], axis=1) for idxs in groups.values()])
        ci_region = bootstrap_ci(weighted_nanmean(W_pairs, J_region))
        for i, grp in enumerate(groups):
            out["region"][(f"ses-{s}", grp)] = (float(np.nanmean(J_region[:, i])), float(ci_region[0, i]), float(ci_region[1, i]))
            vals = r["pair_J"][:, groups[grp]]
            n_def = int(np.sum(~np.isnan(vals)))
            for th in THRESHOLDS:
                pct = 100.0 * float(np.sum(vals > th)) / n_def if n_def else np.nan
                out["exceed"][(f"ses-{s}", grp, th)] = (pct, n_def, vals.size)
        # Comparison B (Phase 3)
        diff_rec = r["W_rec"] - r["O_rec"]
        diff_elem = r["W_elem"] - r["O_elem"]
        g[f"W_{s}"] = float(np.nanmean(r["W_rec"]))
        g[f"O_{s}"] = float(np.nanmean(r["O_rec"]))
        g[f"W_{s} - O_{s}"] = float(np.nanmean(diff_rec))
        g[f"p W_{s} - O_{s}"] = float(bootstrap_p(weighted_nanmean(counts, diff_rec)))
        p_wo = bootstrap_p(weighted_nanmean(counts, diff_elem))
        e[f"W_{s}(e) - O_{s}(e)"] = np.nanmean(diff_elem, axis=0)
        e[f"p W_{s}(e) - O_{s}(e)"] = p_wo
        e[f"q W_{s}(e) - O_{s}(e)"] = np.asarray(multipletests(p_wo, method="fdr_bh")[1])
    g["Delta U"] = g["U_2"] - g["U_1"]
    g["p Delta U"] = float(bootstrap_p(boot_S[2] - boot_S[1]))
    p_du = bootstrap_p(boot_J[2] - boot_J[1])
    e["Delta U(e)"] = e["U_2(e)"] - e["U_1(e)"]
    e["p Delta U(e)"] = p_du
    e["q Delta U(e)"] = np.asarray(multipletests(p_du, method="fdr_bh")[1])
    # Comparisons C and D (Phase 4)
    for key in ("I_self", "I_others", "I_diff", "acc_forward", "acc_backward", "acc_mean"):
        g[key] = res[key]
    for key in ("I_diff", "acc_forward", "acc_backward", "acc_mean"):
        g[f"p {key}"] = res[f"p_{key}"]
    e["I_diff(e)"] = res["I_diff_elem"]
    e["p I_diff(e)"] = res["p_I_diff_elem"]
    e["q I_diff(e)"] = np.asarray(multipletests(res["p_I_diff_elem"], method="fdr_bh")[1])
    # Step 5a classes (Phase 5): universal = U_s(e) significant at both sessions, individual = I_diff(e) significant
    if "q U_1(e)" in e:
        sig_u = (e["q U_1(e)"] < 0.05) & (e["q U_2(e)"] < 0.05)
        sig_i = e["q I_diff(e)"] < 0.05
        e["Class"] = np.where(sig_u & sig_i, CLASSES[2], np.where(sig_u, CLASSES[0], np.where(sig_i, CLASSES[1], CLASSES[3])))
    return out


# ----------------------------------------------------------------------------------------------
# Check against the saved pipeline results (0/0 = NaN reproduces Phases 2-4)
# ----------------------------------------------------------------------------------------------

def validate(fid: str, summ: dict, res: dict, with_shuffle: bool, saved: dict) -> list[dict]:
    """Largest absolute difference between the 0/0 = NaN results here and the saved Phase 2-4 results."""
    u, w, i = (saved[name][fid] for name in ("universality", "within_session", "identifiability"))
    g, e = summ["global"], summ["elem"]
    rows = []

    def add(quantity, ours, saved):
        ours, saved = np.atleast_1d(np.asarray(ours, float)), np.atleast_1d(np.asarray(saved, float))
        both_nan = np.isnan(ours) & np.isnan(saved)
        diff = np.where(both_nan, 0.0, np.abs(ours - saved))
        rows.append({"Feature ID": fid, "Quantity": quantity, "Values compared": int(ours.size),
                     "Max |difference|": float(np.nanmax(np.where(np.isnan(diff), np.inf, diff))),
                     "Values differing by > 1e-9": int(np.sum(~(diff <= 1e-9)))})

    for s in (1, 2):
        ses = f"ses-{s}"
        add(f"J(e) of every pair, {ses}", res[ses]["pair_J"], np.load(OUTPUT_TENSORS_DIR / f"J_pair_{fid}_{ses}.npy"))
        add(f"U_{s}", g[f"U_{s}"], u[ses]["U_global"])
        add(f"U_{s}(e)", e[f"U_{s}(e)"], u[ses]["U_elem"])
        if with_shuffle:
            add(f"p U_{s}(e) (element-shuffle test, drawn differently)", e[f"p U_{s}(e)"], u[ses]["p_elem"])
            sig_ours, sig_saved = e[f"q U_{s}(e)"] < 0.05, np.array(u[ses]["q_elem"]) < 0.05
            rows.append({"Feature ID": fid, "Quantity": f"U_{s}(e) significant (q < 0.05) disagrees", "Values compared": int(sig_ours.size),
                         "Max |difference|": float(np.mean(sig_ours != sig_saved)), "Values differing by > 1e-9": int(np.sum(sig_ours != sig_saved))})
        regions = list(summ["region"])
        add(f"U_{s}(g) and its 95% CI", [summ["region"][k] for k in regions if k[0] == ses],
            [[u[ses]["regional_U"][k[1]], *u[ses]["regional_U_ci_95"][k[1]]] for k in regions if k[0] == ses])
        add(f"W_{s}, O_{s}, W_{s} - O_{s}, p", [g[f"W_{s}"], g[f"O_{s}"], g[f"W_{s} - O_{s}"], g[f"p W_{s} - O_{s}"]],
            [w[ses]["W_global"], w[ses]["O_global"], w[ses]["diff_within_global"], w[ses]["diff_within_p"]])
        add(f"W_{s}(e) - O_{s}(e)", e[f"W_{s}(e) - O_{s}(e)"], w[ses]["diff_within_elem"])
        add(f"p W_{s}(e) - O_{s}(e)", e[f"p W_{s}(e) - O_{s}(e)"], w[ses]["diff_within_elem_p"])
    dev = u["developmental_shift"]
    add("Delta U, p", [g["Delta U"], g["p Delta U"]], [dev["delta_U_global"], dev["p_value"]])
    add("Delta U(e)", e["Delta U(e)"], dev["delta_U_elem"])
    add("p Delta U(e)", e["p Delta U(e)"], dev["delta_U_elem_p"])
    add("I_self, I_others, I_diff", [g["I_self"], g["I_others"], g["I_diff"]], [i["I_self"], i["I_others"], i["I_diff"]])
    add("Accuracy forward, backward, mean", [g["acc_forward"], g["acc_backward"], g["acc_mean"]],
        [i["acc_forward"], i["acc_backward"], i["acc_mean"]])
    add("p I_diff and p accuracy", [g["p I_diff"], g["p acc_forward"], g["p acc_backward"], g["p acc_mean"]],
        [i["p_idiff"], i["p_acc_forward"], i["p_acc_backward"], i["p_acc_mean"]])
    add("I_diff(e)", e["I_diff(e)"], i["I_diff_elem"])
    add("p I_diff(e) (saved run predates the tie tolerance)", e["p I_diff(e)"], i["p_elem"])
    return rows


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def threshold_page(fid: str, exceed: dict):
    """
    Threshold exceedance under both rules: for each session (columns) and threshold (rows), the percentage of the
    J values of each region (pair) above the threshold, 0/0 = NaN and 0/0 = 0 side by side, value on each bar.
    """
    ftype = FEATURE_DEFS[fid]["type"]
    groups = univ.region_groups(ftype)
    vertical = ftype == "edge"
    fig, axes = plt.subplots(len(THRESHOLDS), 2, figsize=(26 if vertical else 18, 5.2 * len(THRESHOLDS) + 2), layout="constrained")
    x = np.arange(len(groups))
    width = 0.4
    for col, ses in enumerate(SESSIONS):
        for row, th in enumerate(THRESHOLDS):
            ax = axes[row, col]
            for offset, rule in zip((-width / 2, width / 2), RULES):
                vals = np.array([exceed[rule][(ses, grp, th)][0] for grp in groups])
                ax.bar(x + offset, vals, width, color=RULE_COLOR[rule])
                for xi, v in zip(x + offset, vals):
                    ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=11, rotation=90 if vertical else 0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{grp} (n = {len(idx)})" if vertical else f"{grp}\n(n = {len(idx)})" for grp, idx in groups.items()],
                               rotation=90 if vertical else 0, fontsize=12)
            ax.set_ylim(0, 125 if vertical else 110)
            ax.set_yticks(range(0, 101, 20))
            ax.tick_params(axis="y", labelsize=12)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.set_ylabel("% of J values above", fontsize=13, fontweight="bold")
            ax.set_title(f"{ses}: J > {th:.2f}", fontsize=15, fontweight="bold")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RULE_COLOR[r]) for r in RULES],
               labels=["0/0 = NaN: 0/0 comparisons left out (main analysis)", "0/0 = 0: 0/0 comparisons counted as similarity 0"],
               loc="outside lower center", ncol=2, fontsize=14)
    fig.suptitle(f"Threshold exceedance under both 0/0 rules: {figs.feature_title(fid)}\n"
                 f"Percentage of the J values of each {'region pair' if vertical else 'region'} (171 infant pairs x its elements) "
                 f"above the threshold; n = elements", fontsize=18, fontweight="bold")
    return fig


def scatter_page(metric: str, fids: list, summaries: dict):
    """One panel per feature: each element's value under 0/0 = 0 against 0/0 = NaN, coloured by which rule finds it significant."""
    n_cols = 4
    n_rows = int(np.ceil(len(fids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 6.0 * n_rows), layout="constrained")
    status_style = [("Significant under both rules", "#2f855a"), ("Only with 0/0 = NaN", "#2b6cb0"),
                    ("Only with 0/0 = 0", "#dd8452"), ("Neither", "#a0aec0")]
    for ax, fid in zip(axes.flat, fids):
        en, ez = summaries[fid]["nan"]["elem"], summaries[fid]["zero"]["elem"]
        if metric == "U":
            xv, yv = (en["U_1(e)"] + en["U_2(e)"]) / 2.0, (ez["U_1(e)"] + ez["U_2(e)"]) / 2.0
            sig_n = (en["q U_1(e)"] < 0.05) & (en["q U_2(e)"] < 0.05)
            sig_z = (ez["q U_1(e)"] < 0.05) & (ez["q U_2(e)"] < 0.05)
        else:
            xv, yv = en["I_diff(e)"], ez["I_diff(e)"]
            sig_n, sig_z = en["q I_diff(e)"] < 0.05, ez["q I_diff(e)"] < 0.05
        status = np.select([sig_n & sig_z, sig_n & ~sig_z, ~sig_n & sig_z], [0, 1, 2], default=3)
        for k, (label, color) in enumerate(status_style):
            m = status == k
            ax.scatter(xv[m], yv[m], s=28, color=color, edgecolors="none", alpha=0.85, label=f"{label} ({m.sum()})")
        lo, hi = np.nanmin([np.nanmin(xv), np.nanmin(yv)]), np.nanmax([np.nanmax(xv), np.nanmax(yv)])
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#444444", linestyle="--", linewidth=1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(figs.feature_title(fid), fontsize=13, fontweight="bold")
        name = r"$U(e) = (U_1(e) + U_2(e)) / 2$" if metric == "U" else r"$I_{\mathrm{diff}}(e)$"
        ax.set_xlabel(f"{name}, 0/0 = NaN", fontsize=12)
        ax.set_ylabel(f"{name}, 0/0 = 0", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=10, loc="upper left", title="Elements (count)", title_fontsize=10)
    for ax in list(axes.flat)[len(fids):]:
        ax.axis("off")
    what = ("universality U(e); significant = universal at both sessions (q < 0.05)" if metric == "U"
            else "individuality I_diff(e); significant = q < 0.05")
    fig.suptitle(f"Every element under both 0/0 rules: {what}\nDashed line: same value under both rules",
                 fontsize=18, fontweight="bold")
    return fig


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------

def wide(rows_by_rule: dict, keys: list, index: dict) -> dict:
    """index columns, then for every key the value under each rule and, for values (not p, q or class), the difference."""
    row = dict(index)
    for key in keys:
        vn, vz = rows_by_rule["nan"].get(key), rows_by_rule["zero"].get(key)
        row[f"{key} [{RULE_LABEL['nan']}]"] = vn
        row[f"{key} [{RULE_LABEL['zero']}]"] = vz
        if not key.startswith(("p ", "q ")) and key != "Class":
            row[f"{key} difference (0/0 = 0 minus 0/0 = NaN)"] = vz - vn
    return row


def main(n_permutations: int, n_workers: int):
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fids = list(FEATURE_DEFS)
    X_all = {fid: np.load(OUTPUT_TENSORS_DIR / f"X_{fid}.npy") for fid in fids}
    affected = [fid for fid in fids if np.any(X_all[fid] == 0)]  # without zeros there is no 0/0
    print(f"0/0 rule comparison | B = {n_permutations} | seed {SEED} | workers {n_workers}")
    print(f"Features with zero values ({len(affected)}): {', '.join(affected)}; the other {len(fids) - len(affected)} cannot change")

    zero_rows = [{"Feature ID": fid, "Feature Name": FEATURE_DEFS[fid]["name"], "Type": FEATURE_DEFS[fid]["type"],
                  "Network": FEATURE_DEFS[fid]["network"], "Has zero values": fid in affected, **zero_counts(X_all[fid])}
                 for fid in fids]
    pd.DataFrame(zero_rows).to_csv(RESULTS_DIR / "zero_counts.csv", index=False)

    # Shuffle tasks (the slow ones) first, so the pool stays busy
    tasks = [(fid, rule, n_permutations, fid in affected) for fid in fids for rule in RULES]
    tasks.sort(key=lambda t: (not t[3], -FEATURE_DEFS[t[0]]["dim"]))
    results = {fid: {} for fid in fids}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for k, (fid, rule, res) in enumerate(pool.map(rule_results, tasks), 1):
            results[fid][rule] = res
            print(f"  [{k:02d}/{len(tasks)}] {fid} {RULE_LABEL[rule]:9} done ({(time.time() - t_start) / 60:.1f} min)", flush=True)

    counts = bootstrap_counts(bootstrap_draws(N, n_permutations, SEED), N)
    W_pairs = counts[:, P_IDX] * counts[:, Q_IDX]
    saved = {}
    for name in ("universality", "within_session", "identifiability"):
        with open(OUTPUT_TENSORS_DIR / f"{name}_results.json") as f:
            saved[name] = json.load(f)
    summaries, validation = {}, []
    for fid in fids:
        summaries[fid] = {rule: summarise(fid, rule, results[fid][rule], counts, W_pairs) for rule in RULES}
        validation += validate(fid, summaries[fid]["nan"], results[fid]["nan"], fid in affected, saved)
        if fid not in affected:  # no zero values: both rules must give the same numbers
            gn, gz = summaries[fid]["nan"]["global"], summaries[fid]["zero"]["global"]
            assert all(gn[k] == gz[k] for k in gn), fid
            en, ez = summaries[fid]["nan"]["elem"], summaries[fid]["zero"]["elem"]
            assert all(np.array_equal(en[k], ez[k], equal_nan=True) for k in en), fid
        print(f"  {fid} summarised ({(time.time() - t_start) / 60:.1f} min)", flush=True)
    del W_pairs, counts

    # Recording-level results, all 26 features
    global_keys = ["U_1", "U_2", "Delta U", "p Delta U", "W_1", "O_1", "W_1 - O_1", "p W_1 - O_1", "W_2", "O_2", "W_2 - O_2",
                   "p W_2 - O_2", "I_self", "I_others", "I_diff", "p I_diff", "acc_forward", "p acc_forward",
                   "acc_backward", "p acc_backward", "acc_mean", "p acc_mean"]
    global_rows = []
    for fid in fids:
        by_rule = {rule: summaries[fid][rule]["global"] for rule in RULES}
        row = wide(by_rule, global_keys, {"Feature ID": fid, "Feature Name": FEATURE_DEFS[fid]["name"], "Has zero values": fid in affected})
        row["Any recording-level value changed"] = any(by_rule["nan"][k] != by_rule["zero"][k] for k in global_keys)
        global_rows.append(row)
    pd.DataFrame(global_rows).to_csv(RESULTS_DIR / "global_comparison.csv", index=False)

    # Element-level results, features with zeros
    elem_keys = ["U_1(e)", "p U_1(e)", "q U_1(e)", "U_2(e)", "p U_2(e)", "q U_2(e)", "Delta U(e)", "p Delta U(e)", "q Delta U(e)",
                 "W_1(e) - O_1(e)", "p W_1(e) - O_1(e)", "q W_1(e) - O_1(e)", "W_2(e) - O_2(e)", "p W_2(e) - O_2(e)",
                 "q W_2(e) - O_2(e)", "I_diff(e)", "p I_diff(e)", "q I_diff(e)", "Class"]
    elem_rows, summary_rows = [], []
    for fid in affected:
        labels, groups, _within, _order = figs.element_info(fid)
        hemi = figs.hemi_category(fid)
        sn, sz = summaries[fid]["nan"]["elem"], summaries[fid]["zero"]["elem"]
        # How common the element is: % of the 19 x 12 infant-epochs of a session where its value is not 0
        present = [100.0 * np.mean(X_all[fid][:, s] != 0, axis=(0, 1)) for s in range(len(SESSIONS))]
        for idx, label in enumerate(labels):
            by_rule = {rule: {k: (v[idx].item() if hasattr(v[idx], "item") else v[idx])
                              for k, v in summaries[fid][rule]["elem"].items()} for rule in RULES}
            row = wide(by_rule, elem_keys, {"Feature ID": fid, "Feature Name": FEATURE_DEFS[fid]["name"], "Element": label,
                                            "Region": groups[idx], "Hemisphere": hemi[idx],
                                            "Present (%), ses-1": float(present[0][idx]), "Present (%), ses-2": float(present[1][idx]),
                                            "Pairs with J(e) undefined, ses-1": int(sn["Pairs with J(e) undefined, ses-1"][idx]),
                                            "Pairs with J(e) undefined, ses-2": int(sn["Pairs with J(e) undefined, ses-2"][idx])})
            row["Class changed"] = by_rule["nan"]["Class"] != by_rule["zero"]["Class"]
            elem_rows.append(row)
        srow = {"Feature ID": fid, "Feature Name": FEATURE_DEFS[fid]["name"], "Elements": len(labels)}
        for rule, e in (("nan", sn), ("zero", sz)):
            tag = f"[{RULE_LABEL[rule]}]"
            srow[f"Mean U_1(e) {tag}"] = float(np.nanmean(e["U_1(e)"]))
            srow[f"Mean U_2(e) {tag}"] = float(np.nanmean(e["U_2(e)"]))
            srow[f"U_1(e) significant {tag}"] = int(np.sum(e["q U_1(e)"] < 0.05))
            srow[f"U_2(e) significant {tag}"] = int(np.sum(e["q U_2(e)"] < 0.05))
            srow[f"Delta U(e) significant, up {tag}"] = int(np.sum((e["q Delta U(e)"] < 0.05) & (e["Delta U(e)"] > 0)))
            srow[f"Delta U(e) significant, down {tag}"] = int(np.sum((e["q Delta U(e)"] < 0.05) & (e["Delta U(e)"] < 0)))
            srow[f"W_1(e) - O_1(e) significant {tag}"] = int(np.sum(e["q W_1(e) - O_1(e)"] < 0.05))
            srow[f"W_2(e) - O_2(e) significant {tag}"] = int(np.sum(e["q W_2(e) - O_2(e)"] < 0.05))
            srow[f"I_diff(e) significant {tag}"] = int(np.sum(e["q I_diff(e)"] < 0.05))
            for cls in CLASSES:
                srow[f"{cls} {tag}"] = int(np.sum(e["Class"] == cls))
        srow["Elements changing class"] = int(np.sum(sn["Class"] != sz["Class"]))
        summary_rows.append(srow)
    pd.DataFrame(elem_rows).to_csv(RESULTS_DIR / "element_comparison.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "element_summary.csv", index=False)

    # Regional universality and threshold exceedance, features with zeros
    region_rows, exceed_rows = [], []
    for fid in affected:
        rn, rz = summaries[fid]["nan"], summaries[fid]["zero"]
        for (ses, grp), (un, lon, hin) in rn["region"].items():
            uz, loz, hiz = rz["region"][(ses, grp)]
            region_rows.append({"Feature ID": fid, "Feature Name": FEATURE_DEFS[fid]["name"], "Session": ses, "Region": grp,
                                f"U_s(g) [{RULE_LABEL['nan']}]": un, f"95% CI [{RULE_LABEL['nan']}]": f"[{lon:.4f}, {hin:.4f}]",
                                f"U_s(g) [{RULE_LABEL['zero']}]": uz, f"95% CI [{RULE_LABEL['zero']}]": f"[{loz:.4f}, {hiz:.4f}]",
                                "Difference (0/0 = 0 minus 0/0 = NaN)": uz - un})
        for (ses, grp, th), (pn, n_def, n_all) in rn["exceed"].items():
            pz = rz["exceed"][(ses, grp, th)][0]
            exceed_rows.append({"Feature ID": fid, "Feature Name": FEATURE_DEFS[fid]["name"], "Session": ses, "Region": grp,
                                "Threshold": th, "J values in group": n_all, "J values defined (not 0/0)": n_def,
                                f"% above threshold [{RULE_LABEL['nan']}]": pn, f"% above threshold [{RULE_LABEL['zero']}]": pz,
                                "Difference (0/0 = 0 minus 0/0 = NaN)": pz - pn})
    pd.DataFrame(region_rows).to_csv(RESULTS_DIR / "regional_comparison.csv", index=False)
    pd.DataFrame(exceed_rows).to_csv(RESULTS_DIR / "threshold_comparison.csv", index=False)
    pd.DataFrame(validation).to_csv(RESULTS_DIR / "validation_vs_pipeline.csv", index=False)

    # Figures
    exceed = {fid: {rule: summaries[fid][rule]["exceed"] for rule in RULES} for fid in affected}
    with PdfPages(FIG_DIR / "threshold_bars_nan_vs_zero.pdf") as pdf:
        for fid in affected:
            fig = threshold_page(fid, exceed[fid])
            pdf.savefig(fig)
            if fid == "f10":
                fig.savefig(FIG_DIR / "threshold_bars_nan_vs_zero_f10.png", dpi=110)
            plt.close(fig)
    with PdfPages(FIG_DIR / "element_values_nan_vs_zero.pdf") as pdf:
        for metric, png in (("U", "element_U_nan_vs_zero.png"), ("I_diff", "element_I_diff_nan_vs_zero.png")):
            fig = scatter_page(metric, affected, summaries)
            pdf.savefig(fig)
            fig.savefig(FIG_DIR / png, dpi=110)
            plt.close(fig)

    with open(RESULTS_DIR / "run_info.json", "w") as f:
        json.dump({"n_permutations": n_permutations, "seed": SEED, "features_with_zero_values": affected,
                   "minutes": round((time.time() - t_start) / 60, 1),
                   "note": "0/0 = NaN results recomputed with the current pipeline code; see validation_vs_pipeline.csv"}, f, indent=2)
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Results: {RESULTS_DIR} | Figures: {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS, help="Permutations and bootstrap resamples (default: config.N_PERMUTATIONS)")
    parser.add_argument("--workers", type=int, default=8, help="Worker processes (default: 8)")
    args = parser.parse_args()
    main(args.permutations, args.workers)
