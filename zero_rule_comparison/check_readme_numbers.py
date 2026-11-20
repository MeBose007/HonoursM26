"""
Checks every number in zero_rule_comparison/README.md against the result tables.

Each sentence of the README that states a result is rebuilt here from results/*.csv (and the run files) and must
appear in the README word for word. Every other number in the README must belong to a fixed part of the method
(listed in METHOD_TEXT), so no number can appear without a check.
Run from the project root after compare_zero_rules.py and check_comparison.py:
.venv/bin/python zero_rule_comparison/check_readme_numbers.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
README = (HERE / "README.md").read_text()
N, Z = "[0/0 = NaN]", "[0/0 = 0]"
DIFF = "difference (0/0 = 0 minus 0/0 = NaN)"

z = pd.read_csv(RES / "zero_counts.csv").set_index("Feature ID")
g = pd.read_csv(RES / "global_comparison.csv").set_index("Feature ID")
e = pd.read_csv(RES / "element_comparison.csv")
s = pd.read_csv(RES / "element_summary.csv").set_index("Feature ID")
r = pd.read_csv(RES / "regional_comparison.csv")
t = pd.read_csv(RES / "threshold_comparison.csv")
v = pd.read_csv(RES / "validation_vs_pipeline.csv")
run = json.loads((RES / "run_info.json").read_text())
report = (RES / "check_report.txt").read_text()
problems = []


def need(condition, what):
    if not condition:
        problems.append(f"Fact does not hold: {what}")


def el(fid, element):
    return e[(e["Feature ID"] == fid) & (e["Element"] == element)].iloc[0]


def pct(x):
    return f"{x:.1f}"


affected = list(s.index)
unaffected = [f for f in g.index if f not in affected]
changed = list(g.index[g["Any recording-level value changed"]])
tot = s.sum(numeric_only=True)


def total(name, rule):
    return int(tot[f"{name} [{rule}]"])


du_n = total("Delta U(e) significant, up", "0/0 = NaN") + total("Delta U(e) significant, down", "0/0 = NaN")
du_z = total("Delta U(e) significant, up", "0/0 = 0") + total("Delta U(e) significant, down", "0/0 = 0")
sig_u_n = (e[f"q U_1(e) {N}"] < 0.05) & (e[f"q U_2(e) {N}"] < 0.05)
sig_u_z = (e[f"q U_1(e) {Z}"] < 0.05) & (e[f"q U_2(e) {Z}"] < 0.05)
lost_u, gained_u = e[sig_u_n & ~sig_u_z], e[~sig_u_n & sig_u_z]
dq_n, dq_z = e[f"q Delta U(e) {N}"] < 0.05, e[f"q Delta U(e) {Z}"] < 0.05
presence_change = (e["Present (%), ses-2"] - e["Present (%), ses-1"]).abs()
binary_links = [f for f in affected if f not in ("f02", "f03", "f11")]
zero_cols = ["Epoch comparisons 0/0 ses-1 (%)", "Epoch comparisons 0/0 ses-2 (%)", "Epoch comparisons 0/0 cross-session (%)"]


def zero_range(fids):
    vals = z.loc[fids, zero_cols].to_numpy()
    return f"{vals.min():.0f}–{vals.max():.0f}%"


f02 = g.loc["f02"]
f02_values = [c for c in g.columns if c.endswith(DIFF)]
fp = el("f10", "Fp1-O2")
fz = el("f26", "FZ-P3")
f20 = el("f20", "F8-T6")
a, b = el("f12", "Fp2-F7"), el("f12", "C4-P4")
cls_changes = e[e["Class changed"]]
rd = r["Difference (0/0 = 0 minus 0/0 = NaN)"]
r_min = r.loc[rd.idxmin()]
td = t["Difference (0/0 = 0 minus 0/0 = NaN)"]
t_min = t.loc[td.idxmin()]


def trow(fid, ses, region, th):
    return t[(t["Feature ID"] == fid) & (t["Session"] == ses) & (t["Region"] == region) & (t["Threshold"] == th)].iloc[0]


fc20, ff10, fc10 = trow("f20", "ses-2", "F-C", 0.3), trow("f10", "ses-1", "F-F", 0.3), trow("f10", "ses-1", "F-C", 0.3)
t2 = pd.read_csv(HERE.parent / "tables" / "Table_T2_Element_Results.csv")
pdf_pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(HERE / "figures" / "threshold_bars_nan_vs_zero.pdf")],
                                                             capture_output=True, text=True).stdout).group(1))
n_checked = int(re.search(r"Values checked: (\d+)", report).group(1))
n_report_problems = int(re.search(r"Problems: (\d+)", report).group(1))

# Conditions behind the sentences (beyond the numbers they quote)
need(len(changed) == 1 and changed == ["f02"], "only f02 has a recording-level change")
need(max(abs(f02[c]) for c in f02_values) < 1e-4, "every f02 value changes by less than 0.0001")
need(td.max() <= 0 and rd.max() <= 0, "percentages and U_s(g) are never higher under 0/0 = 0")
need((e[f"U_1(e) {DIFF}"] <= 0).all() and (e[f"U_2(e) {DIFF}"] <= 0).all(), "U(e) is never higher under 0/0 = 0")
need(t_min["Threshold"] == 0.3, "the largest threshold drop is at 0.30")
need(r_min["Feature ID"] == "f26" and r_min["Session"] == "ses-2" and r_min["Region"] == "T-T", "largest U_s(g) drop is f26 ses-2 T-T")
need(t_min["Feature ID"] == "f24" and t_min["Session"] == "ses-2" and t_min["Region"] == "T-T", "largest threshold drop is f24 ses-2 T-T")
need(fc20["J values defined (not 0/0)"] == fc20["J values in group"], "f20 ses-2 F-C has no undefined value")
need(len(gained_u) == 1 and gained_u.iloc[0]["Feature ID"] == "f03" and gained_u.iloc[0]["Element"] == "F3-C4", "f03 F3-C4 is the only gain")
need(z[["S epoch pairs 0/0 ses-1", "S epoch pairs 0/0 ses-2"]].to_numpy().sum() == 0
     and z["S epoch pairs 0/0 cross-session"].sum() == z.loc["f02", "S epoch pairs 0/0 cross-session"], "all whole-epoch 0/0 are in f02, cross-session")
for ses in (1, 2):
    only_zero = (e[f"q W_{ses}(e) - O_{ses}(e) {Z}"] < 0.05) & ~(e[f"q W_{ses}(e) - O_{ses}(e) {N}"] < 0.05)
    need(only_zero.sum() == 0, f"no W_{ses}(e) - O_{ses}(e) is significant only under 0/0 = 0")
for fid in ("f10", "f12", "f14", "f16", "f18"):
    x = el(fid, "Fp1-O2")
    need(x[f"q W_2(e) - O_2(e) {N}"] < 0.05 <= x[f"q W_2(e) - O_2(e) {Z}"], f"{fid} Fp1-O2 W_2 - O_2 significant only under 0/0 = NaN")
need(fz[f"q W_2(e) - O_2(e) {N}"] < 0.05 <= fz[f"q W_2(e) - O_2(e) {Z}"], "f26 FZ-P3 significant only under 0/0 = NaN")
need(f20[f"q I_diff(e) {N}"] < 0.05 <= f20[f"q I_diff(e) {Z}"], "f20 F8-T6 significant only under 0/0 = NaN")
need(all(x[f"q I_diff(e) {rule}"] < 0.05 for x in (a, b) for rule in (N, Z)), "f12 Fp2-F7 and C4-P4 significant under both rules")
need(set((e["Feature ID"] + " " + e["Element"])[(e[f"q I_diff(e) {N}"] < 0.05) | (e[f"q I_diff(e) {Z}"] < 0.05)]) ==
     {"f12 Fp2-F7", "f12 C4-P4", "f20 F8-T6"}, "the three significant I_diff(e) links")
need(run["n_permutations"] == 200000 and run["seed"] == 42, "200,000 permutations and seed 42")
need(20 <= run["minutes"] <= 40, "run time about 30 minutes")
need(abs(fp["Present (%), ses-2"] * 228 / 100 - 2) < 1e-9, "f10 Fp1-O2 present in 2 of 228 baby-epochs at ses-2")
other_diffs = v[(v["Values differing by > 1e-9"] > 0) & ~v["Quantity"].str.contains("drawn differently|predates")]
need(len(other_diffs) == 1 and other_diffs.iloc[0]["Feature ID"] == "f19" and other_diffs.iloc[0]["Values differing by > 1e-9"] == 1,
     "apart from I_diff(e) p, only one f19 accuracy p differs")
need(np.isclose(2 / 3, 0.67, atol=0.005) and np.isclose(2 / 12, 0.17, atol=0.005), "the worked example")

# Sentences that state results, rebuilt from the tables
RESULT_TEXT = [
    f"That happens in {len(affected)} of the 26 feature sets",
    f"The other {len(unaffected)} feature sets give identical results under both rules.",
    f"are identical for {len(g) - len(changed)} feature sets. In f02 they change by less than 0.0001.",
    f"universal ({len(e[sig_u_n])} → {len(e[sig_u_z])})",
    f"between 3 and 6 months ({du_n} → {du_z})",
    f"by up to {-td.min():.1f} percentage points, mostly at the 0.30 threshold",
    f"for those {len(affected)} feature sets:",
    f"There are {len(affected)}:",
    f"The other {len(unaffected)} feature sets have no zero values at all: {', '.join(unaffected[:-1])} and {unaffected[-1]}.",
    f"of the {len(affected)} feature sets with zeros",
    f"| The nine binary link features (f10–f26) | {zero_range(binary_links)} |",
    f"| f02, weighted node betweenness | {zero_range(['f02'])} |",
    f"| f03, weighted edge betweenness | {zero_range(['f03'])} |",
    f"| f11, binary node betweenness | {zero_range(['f11'])} |",
    f"That happened in just {int(z['S epoch pairs 0/0 cross-session'].sum())} epoch pairs in the whole dataset",
    f"for {len(g) - len(changed)} of the 26 feature sets",
    f"caused by those {int(z.loc['f02', 'S epoch pairs 0/0 cross-session'])} all-zero epoch pairs",
    f"| I_others | {f02[f'I_others {N}']:.6f} | {f02[f'I_others {Z}']:.6f} |",
    f"| I_diff | {f02[f'I_diff {N}']:.6f} | {f02[f'I_diff {Z}']:.6f} |",
    f"| p of I_diff | {f02[f'p I_diff {N}']:.4f} | {f02[f'p I_diff {Z}']:.4f} |",
    f"| Identification accuracy | {f02[f'acc_mean {N}']:.1f}% | {f02[f'acc_mean {Z}']:.1f}% |",
    f"Counts over the {len(affected)} feature sets that contain zeros, {int(tot['Elements']):,} nodes and links in total.",
    f"| Universal at ses-1, U_1(e) significant | {total('U_1(e) significant', '0/0 = NaN')} | {total('U_1(e) significant', '0/0 = 0')} |",
    f"| Universal at ses-2, U_2(e) significant | {total('U_2(e) significant', '0/0 = NaN')} | {total('U_2(e) significant', '0/0 = 0')} |",
    f"| Universal at both sessions | {len(e[sig_u_n])} | {len(e[sig_u_z])} |",
    f"| Significant change 3 → 6 months, ΔU(e) up | {total('Delta U(e) significant, up', '0/0 = NaN')} | {total('Delta U(e) significant, up', '0/0 = 0')} |",
    f"| Significant change 3 → 6 months, ΔU(e) down | {total('Delta U(e) significant, down', '0/0 = NaN')} | {total('Delta U(e) significant, down', '0/0 = 0')} |",
    f"| Stable within a session, W_1(e) − O_1(e) significant | {total('W_1(e) - O_1(e) significant', '0/0 = NaN')} | {total('W_1(e) - O_1(e) significant', '0/0 = 0')} |",
    f"| Stable within a session, W_2(e) − O_2(e) significant | {total('W_2(e) - O_2(e) significant', '0/0 = NaN')} | {total('W_2(e) - O_2(e) significant', '0/0 = 0')} |",
    f"| Personal across sessions, I_diff(e) significant | {total('I_diff(e) significant', '0/0 = NaN')} | {total('I_diff(e) significant', '0/0 = 0')} |",
    f"{len(lost_u)} links that are universal under 0/0 = NaN are not universal under 0/0 = 0, and {len(gained_u)} link gains (f03, F3–C4).",
    f"({du_z} against {du_n})",
    f"by {presence_change[~dq_n & dq_z].mean():.1f} percentage points, compared with {presence_change[~dq_n & ~dq_z].mean():.1f} points",
    f"present in only {round(fp['Present (%), ses-2'] * 228 / 100)} of the 228 baby-epochs ({pct(fp['Present (%), ses-2'])}%)",
    f"undefined for {int(fp['Pairs with J(e) undefined, ses-2'])} of the 171 baby pairs",
    f"W_2(e) − O_2(e) = {fp[f'W_2(e) - O_2(e) {N}']:.3f}, the largest possible value, with q = {fp[f'q W_2(e) - O_2(e) {N}']:.4f}",
    f"it is {fp[f'W_2(e) - O_2(e) {Z}']:.3f}, with q = {fp[f'q W_2(e) - O_2(e) {Z}']:.2f}",
    f"goes from {fz[f'W_2(e) - O_2(e) {N}']:.3f} to {fz[f'W_2(e) - O_2(e) {Z}']:.3f}",
    f"f12, Fp2–F7 (q = {a[f'q I_diff(e) {N}']:.3f} and {a[f'q I_diff(e) {Z}']:.3f}) and f12, C4–P4 (q = {b[f'q I_diff(e) {N}']:.3f} and {b[f'q I_diff(e) {Z}']:.3f})",
    f"I_diff(e) = {f20[f'I_diff(e) {N}']:.3f} under 0/0 = NaN (q = {f20[f'q I_diff(e) {N}']:.3f})",
    f"I_diff(e) = {f20[f'I_diff(e) {Z}']:.3f} under 0/0 = 0 (q = {f20[f'q I_diff(e) {Z}']:.2f})",
    f"undefined for {int(f20['Pairs with J(e) undefined, ses-1'])} baby pairs at ses-1 and {int(f20['Pairs with J(e) undefined, ses-2'])} at ses-2",
    f"{len(cls_changes)} of the {len(e):,} nodes and links change class",
    f"{int(((cls_changes[f'Class {N}'] == 'Universal') & (cls_changes[f'Class {Z}'] == 'Neither')).sum())} go from universal to neither.",
    f"in all {len(r)} feature × session × region combinations. {int((rd == 0).sum())} are unchanged.",
    f"falls from {r_min[f'U_s(g) {N}']:.4f} to {r_min[f'U_s(g) {Z}']:.4f}",
    f"all {int(fc20['J values in group']):,} values are defined, yet the share above 0.30 falls from "
    f"{pct(fc20[f'% above threshold {N}'])}% to {pct(fc20[f'% above threshold {Z}'])}%",
    *[f"| J > {th:.2f} | {-t.loc[t['Threshold'] == th, 'Difference (0/0 = 0 minus 0/0 = NaN)'].mean():.1f} | "
      f"{-t.loc[t['Threshold'] == th, 'Difference (0/0 = 0 minus 0/0 = NaN)'].min():.1f} |" for th in (0.3, 0.5, 0.75, 0.9)],
    f"Averages are over the {int((t['Threshold'] == 0.3).sum())} feature × session × region combinations.",
    f"T-T: {pct(t_min[f'% above threshold {N}'])}% → {pct(t_min[f'% above threshold {Z}'])}% above 0.30",
    f"F-F goes from {pct(ff10[f'% above threshold {N}'])}% to {pct(ff10[f'% above threshold {Z}'])}% above 0.30",
    f"F-C goes from {pct(fc10[f'% above threshold {N}'])}% to {pct(fc10[f'% above threshold {Z}'])}%",
    f"All {pdf_pages} feature sets are in",
    f"Both rules use {run['n_permutations']:,} permutations and bootstrap resamples, seed {run['seed']}",
    f"all {int(v.loc[v['Quantity'].str.startswith('J(e) of every pair'), 'Values compared'].sum()):,} pair-by-element similarities",
    f"All {int(v.loc[v['Quantity'].str.contains('significant'), 'Values compared'].sum()):,} significance decisions for U_1(e) and U_2(e)",
    f"The only differences are {int(v.loc[v['Quantity'].str.contains('predates'), 'Values differing by > 1e-9'].sum())} p-values of I_diff(e)",
    f"the main tables still list {int((t2['q_fdr_idiff'] < 0.05).sum())} significant I_diff(e) links, "
    f"but the current code finds {total('I_diff(e) significant', '0/0 = NaN')}",
    f"({n_checked:,} values)",
    f"It found {n_report_problems} problems",
]
need(int(v.loc[v["Quantity"].str.contains("significant"), "Values differing by > 1e-9"].sum()) == 0, "all U(e) significance decisions agree")
need(int(v.loc[~v["Quantity"].str.contains("drawn differently|predates|p I_diff and p accuracy"), "Values differing by > 1e-9"].sum()) == 0,
     "all other values match the pipeline exactly")

# Fixed parts of the method and the worked example; their numbers are not results
METHOD_TEXT = [
    "0/0", "0/0 = 0", "0/0 = 1", "26 feature sets", "robustness check 1", "exactly 0", "(both babies 0)", "--workers 12",
    "Phase 7's faster version of the Phase 2 shuffle test", "agree with Phase 2", "1. **The count", "2. **The values",
    "compared are 0", "as 0 can only", "now count as 0",
    *[f"## {k}." for k in range(1, 8)], "0.4 and 0.8", "0.4 / 0.8 = 0.5", "1 and 1", "1 / 1 = 1", "1 and 0", "0 / 1 = 0", "0 and 0", "0 / 0",
    "their 12 epochs", "epochs 1, 2 and 3", "epochs 1 and 2", "epochs 4 to 12", "1, 1, 0", "(1 + 1 + 0) / 3", "| 0.67 |",
    "(1 + 1 + 0 + 0 + … + 0) / 12", "| 0.17 |", "12 epochs", "3 and 6 months", "3 → 6 months", "q < 0.05", "J > 0.30",
    "J > 0.50", "J > 0.75", "J > 0.90", "above 0.30", "0.30 threshold", "0.30, 0.50, 0.75 and 0.90", "(0 or 1)", "95% CI",
    "(page 1)", "(page 2)", "Phases 2 to 5", "Phase 7 check 1", "3 November", "about 30 minutes", "171 baby pairs",
    "228 baby-epochs", "less than 0.0001", "of the 26 feature sets", "6 months",
]

# Every number in the README must lie inside a checked result sentence or a fixed part of the method
covered = np.zeros(len(README), dtype=bool)
for text in RESULT_TEXT:
    hits = [m.start() for m in re.finditer(re.escape(text), README)]
    if not hits:
        problems.append(f"Not found in README: {text!r}")
    for h in hits:
        covered[h:h + len(text)] = True
for text in METHOD_TEXT:
    for m in re.finditer(re.escape(text), README):
        covered[m.start():m.end()] = True
for m in re.finditer(r"(?<![\w.\-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)", README):
    if not covered[m.start():m.end()].all():
        line = README.count("\n", 0, m.start()) + 1
        problems.append(f"Unchecked number {m.group(0)!r} on README line {line}")

print(f"Result sentences checked: {len(RESULT_TEXT)} | Problems: {len(problems)}")
for p in problems:
    print("  ", p)
sys.exit(1 if problems else 0)
