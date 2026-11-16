"""
Independent check of the 0/0 rule comparison tables in zero_rule_comparison/results.

Recomputes every value in the tables straight from the feature tensors X_f.npy, with its own short code (no
pipeline functions), under both rules:
  0/0 = NaN: a comparison where both values are 0 is left out of the mean
  0/0 = 0:   it counts as similarity 0
and checks U_s, W_s, O_s, I_self, I_others, I_diff and accuracy (all 26 features), and U_s(e), Delta U(e),
W_s(e) - O_s(e), I_diff(e), U_s(g) and the threshold exceedance (features with zeros). It also recomputes the FDR
q-values from the p-values, the Step 5a classes from the q-values, and every count in element_summary.csv.
The p-values themselves come from permutations and bootstrap resamples; validation_vs_pipeline.csv compares them
with the main analysis.
Run from the project root: .venv/bin/python zero_rule_comparison/check_comparison.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT))
from analysis_pipeline.config import (
    EDGE_LABELS,
    FEATURE_DEFS,
    NODE_LABELS,
    OUTPUT_TENSORS_DIR,
)

TOL = 1e-9
N = 19
PAIRS = list(combinations(range(N), 2))
RULE_TAG = {"nan": "[0/0 = NaN]", "zero": "[0/0 = 0]"}
problems = []
checked = 0


def ratio_mean(x, y, rule, axis=0):
    """Mean of min/max over axis; 0/0 left out (NaN rule) or counted as 0 (zero rule)."""
    mn, mx = np.minimum(x, y), np.maximum(x, y)
    defined = mx > 0
    r = np.where(defined, mn / np.where(defined, mx, 1.0), 0.0)
    if rule == "zero":
        return r.mean(axis=axis)
    n_def = defined.sum(axis=axis)
    return np.where(n_def > 0, r.sum(axis=axis) / np.maximum(n_def, 1), np.nan)


def S_value(x, y, rule):
    """Recording similarity: mean over epoch pairs of sum(min) / sum(max)."""
    num, den = np.minimum(x, y).sum(axis=1), np.maximum(x, y).sum(axis=1)
    ok = den > 0
    vals = num[ok] / den[ok]
    if rule == "zero":
        return vals.sum() / len(den)
    return vals.mean() if ok.any() else np.nan


def mean_ignoring_nan(v, axis=None):
    v = np.asarray(v, float)
    ok = ~np.isnan(v)
    s, c = np.where(ok, v, 0.0).sum(axis=axis), ok.sum(axis=axis)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def accuracy(M):
    """1-NN identification in %: row (forward) and column (backward) maximum must be the diagonal, without a tie."""
    fwd = [M[a].max() == M[a, a] and np.sum(M[a] == M[a].max()) == 1 for a in range(N)]
    bwd = [M[:, a].max() == M[a, a] and np.sum(M[:, a] == M[:, a].max()) == 1 for a in range(N)]
    return 100.0 * np.mean(fwd), 100.0 * np.mean(bwd)


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def compare(what, ours, table):
    global checked
    ours, table = np.asarray(ours, float), np.asarray(table, float)
    bad = ~((np.isnan(ours) & np.isnan(table)) | (np.abs(ours - table) <= TOL))
    checked += ours.size
    if bad.any():
        problems.append(f"{what}: {int(bad.sum())} of {ours.size} differ, e.g. ours {ours[bad][:3]} vs table {table[bad][:3]}")


def region(ch):
    return ch[0]  # Fp1 -> F, T5 -> T


def groups_of(fid):
    if FEATURE_DEFS[fid]["type"] == "node":
        return np.array([region(n) for n in NODE_LABELS])
    order = "FCTPO"
    return np.array(["-".join(sorted((region(a), region(b)), key=order.index)) for a, b in (e.split("-") for e in EDGE_LABELS)])


glob = pd.read_csv(RES / "global_comparison.csv").set_index("Feature ID")
elem = pd.read_csv(RES / "element_comparison.csv")
summ = pd.read_csv(RES / "element_summary.csv").set_index("Feature ID")
reg = pd.read_csv(RES / "regional_comparison.csv")
thr = pd.read_csv(RES / "threshold_comparison.csv")
first, second = np.arange(6), np.arange(6, 12)

for fid in FEATURE_DEFS:
    X = np.load(OUTPUT_TENSORS_DIR / f"X_{fid}.npy")  # (19, 2, 12, dim)
    has_zero = bool(np.any(X == 0))
    if bool(glob.loc[fid, "Has zero values"]) != has_zero:
        problems.append(f"{fid}: 'Has zero values' flag")
    for rule in ("nan", "zero"):
        tag = RULE_TAG[rule]
        g = glob.loc[fid]
        U = {}
        for s in (1, 2):
            S = [S_value(X[a, s - 1], X[b, s - 1], rule) for a, b in PAIRS]
            U[s] = mean_ignoring_nan(S)
            compare(f"{fid} U_{s} {tag}", U[s], g[f"U_{s} {tag}"])
            Xs = X[:, s - 1]
            W = np.array([S_value(Xs[a, first], Xs[a, second], rule) for a in range(N)])
            O = np.array([mean_ignoring_nan([S_value(Xs[a, first], Xs[b, second], rule) for b in range(N) if b != a]) for a in range(N)])
            compare(f"{fid} W_{s}, O_{s}, W_{s} - O_{s} {tag}", [mean_ignoring_nan(W), mean_ignoring_nan(O), mean_ignoring_nan(W - O)],
                    [g[f"W_{s} {tag}"], g[f"O_{s} {tag}"], g[f"W_{s} - O_{s} {tag}"]])
        compare(f"{fid} Delta U {tag}", U[2] - U[1], g[f"Delta U {tag}"])
        M = np.array([[S_value(X[a, 0], X[b, 1], rule) for b in range(N)] for a in range(N)])
        diag = np.eye(N, dtype=bool)
        I_self, I_others = mean_ignoring_nan(M[diag]), mean_ignoring_nan(M[~diag])
        f, b = accuracy(np.where(np.isnan(M), -np.inf, M))
        compare(f"{fid} I_self, I_others, I_diff, accuracy {tag}", [I_self, I_others, I_self - I_others, f, b, (f + b) / 2],
                [g[f"I_self {tag}"], g[f"I_others {tag}"], g[f"I_diff {tag}"], g[f"acc_forward {tag}"], g[f"acc_backward {tag}"], g[f"acc_mean {tag}"]])

        if not has_zero:
            continue
        e = elem[elem["Feature ID"] == fid]
        grp = groups_of(fid)
        if list(e["Region"]) != list(grp):
            problems.append(f"{fid}: region labels")
        pair_J = {}
        for s in (1, 2):
            if rule == "nan":
                compare(f"{fid} present ses-{s}", 100.0 * (X[:, s - 1] != 0).mean(axis=(0, 1)), e[f"Present (%), ses-{s}"])
            pair_J[s] = np.array([ratio_mean(X[a, s - 1], X[b, s - 1], rule) for a, b in PAIRS])  # (171, dim)
            compare(f"{fid} U_{s}(e) {tag}", mean_ignoring_nan(pair_J[s], axis=0), e[f"U_{s}(e) {tag}"])
            if rule == "nan":
                compare(f"{fid} pairs with J(e) undefined ses-{s}", np.isnan(pair_J[s]).sum(axis=0), e[f"Pairs with J(e) undefined, ses-{s}"])
            Xs = X[:, s - 1]
            W_e = np.array([ratio_mean(Xs[a, first], Xs[a, second], rule) for a in range(N)])
            O_e = np.array([mean_ignoring_nan([ratio_mean(Xs[a, first], Xs[b, second], rule) for b in range(N) if b != a], axis=0)
                            for a in range(N)])
            compare(f"{fid} W_{s}(e) - O_{s}(e) {tag}", mean_ignoring_nan(W_e - O_e, axis=0), e[f"W_{s}(e) - O_{s}(e) {tag}"])
            # Regional universality and threshold exceedance
            r = reg[(reg["Feature ID"] == fid) & (reg["Session"] == f"ses-{s}")].set_index("Region")
            t = thr[(thr["Feature ID"] == fid) & (thr["Session"] == f"ses-{s}")]
            for gname in r.index:
                cols = grp == gname
                compare(f"{fid} U_{s}({gname}) {tag}", mean_ignoring_nan(mean_ignoring_nan(pair_J[s][:, cols], axis=1)),
                        r.loc[gname, f"U_s(g) {tag}"])
                vals = pair_J[s][:, cols]
                for _, row in t[t["Region"] == gname].iterrows():
                    n_def = np.sum(~np.isnan(vals))
                    compare(f"{fid} ses-{s} {gname} > {row['Threshold']} {tag}", 100.0 * np.sum(vals > row["Threshold"]) / n_def,
                            row[f"% above threshold {tag}"])
                    if rule == "nan":
                        compare(f"{fid} ses-{s} {gname} value counts", [n_def, vals.size],
                                [row["J values defined (not 0/0)"], row["J values in group"]])
        compare(f"{fid} Delta U(e) {tag}", mean_ignoring_nan(pair_J[2], axis=0) - mean_ignoring_nan(pair_J[1], axis=0), e[f"Delta U(e) {tag}"])
        M_e = np.array([[ratio_mean(X[a, 0], X[b, 1], rule) for b in range(N)] for a in range(N)])  # (19, 19, dim)
        compare(f"{fid} I_diff(e) {tag}", mean_ignoring_nan(M_e[diag], axis=0) - mean_ignoring_nan(M_e[~diag], axis=0), e[f"I_diff(e) {tag}"])

        # q from p (Benjamini-Hochberg across the feature's elements), classes from q, summary counts
        for test in ("U_1(e)", "U_2(e)", "Delta U(e)", "W_1(e) - O_1(e)", "W_2(e) - O_2(e)", "I_diff(e)"):
            compare(f"{fid} q {test} {tag}", bh(e[f"p {test} {tag}"]), e[f"q {test} {tag}"])
        sig_u = (e[f"q U_1(e) {tag}"] < 0.05) & (e[f"q U_2(e) {tag}"] < 0.05)
        sig_i = e[f"q I_diff(e) {tag}"] < 0.05
        cls = np.where(sig_u & sig_i, "Universal and Individual", np.where(sig_u, "Universal", np.where(sig_i, "Individual", "Neither")))
        if not np.array_equal(cls, e[f"Class {tag}"].to_numpy()):
            problems.append(f"{fid} classes {tag}")
        srow = summ.loc[fid]
        counts = {
            f"U_1(e) significant {tag}": (e[f"q U_1(e) {tag}"] < 0.05).sum(),
            f"U_2(e) significant {tag}": (e[f"q U_2(e) {tag}"] < 0.05).sum(),
            f"Delta U(e) significant, up {tag}": ((e[f"q Delta U(e) {tag}"] < 0.05) & (e[f"Delta U(e) {tag}"] > 0)).sum(),
            f"Delta U(e) significant, down {tag}": ((e[f"q Delta U(e) {tag}"] < 0.05) & (e[f"Delta U(e) {tag}"] < 0)).sum(),
            f"W_1(e) - O_1(e) significant {tag}": (e[f"q W_1(e) - O_1(e) {tag}"] < 0.05).sum(),
            f"W_2(e) - O_2(e) significant {tag}": (e[f"q W_2(e) - O_2(e) {tag}"] < 0.05).sum(),
            f"I_diff(e) significant {tag}": sig_i.sum(),
            **{f"{c} {tag}": int((cls == c).sum()) for c in ("Universal", "Individual", "Universal and Individual", "Neither")},
            f"Mean U_1(e) {tag}": e[f"U_1(e) {tag}"].mean(),
            f"Mean U_2(e) {tag}": e[f"U_2(e) {tag}"].mean(),
        }
        compare(f"{fid} summary counts {tag}", list(counts.values()), [srow[k] for k in counts])
    if has_zero:
        e = elem[elem["Feature ID"] == fid]
        changed = e["Class [0/0 = NaN]"] != e["Class [0/0 = 0]"]
        if not np.array_equal(changed.to_numpy(), e["Class changed"].to_numpy()) or changed.sum() != summ.loc[fid, "Elements changing class"]:
            problems.append(f"{fid}: class-change flags or count")

report = [f"Values checked: {checked}", f"Problems: {len(problems)}"] + [f"   {p}" for p in problems[:40]]
print("\n".join(report))
(RES / "check_report.txt").write_text("\n".join(report) + "\n")
sys.exit(1 if problems else 0)
