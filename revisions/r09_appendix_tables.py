"""Revision R9: descriptive statistics and full regression tables for the appendix.

Writes two LaTeX tables, generated from the modelling table of Notebook 05, so that every number can be traced:
  outputs/tables/revisions/r09_descriptive_statistics.tex   variables of the models, on the network films
  outputs/tables/revisions/r09_full_regressions_m4.tex       model M4 for the four main outcomes, all coefficients
                                                             except the year, genre and community indicators
The coefficients of M4 are checked against the published Table 4 before the table is written.

Run from the project root:  python revisions/r09_appendix_tables.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables" / "revisions"
df = pd.read_csv(ROOT / "processed" / "features" / "modeling_table_success_star_power.csv", low_memory=False)
mm = 10_000
ok = (df["revenue_clean"] >= mm) & (df["budget_clean"] >= mm)
df["log_roi"] = np.where(ok, np.log(df["revenue_clean"].where(ok) / df["budget_clean"].where(ok)), np.nan)
tf = pd.read_csv(ROOT / "processed" / "features" / "time_aware_plot_features.csv",
                 usecols=["tmdb_id", "crowding_raw", "foresight_raw", "competition_raw"])
df = df.merge(tf, on="tmdb_id", how="left")

# ── descriptive statistics (unstandardized variables) ────────────────────────────────────────────
VARS = [
    ("Outcomes", None),
    ("Revenue percentile within the year", "revenue_percentile_year"), ("Log revenue", "log_revenue"),
    ("Log return on investment", "log_roi"), ("Rating (0--10)", "vote_average"), ("Log popularity", "log_popularity"),
    ("Network position", None),
    ("Novelty", "novelty"), ("Log betweenness, $\\log(1 + N b)$", "log_betweenness"),
    ("Participation coefficient", "participation"), ("Weighted clustering coefficient", "clustering"),
    ("Position in time (before standardization)", None),
    ("Crowding", "crowding_raw"), ("Same-year competition", "competition_raw"), ("Foresight", "foresight_raw"),
    ("Star power and controls", None),
    ("Cast star index, $\\log(1 + \\text{prior votes})$", "star_index"),
    ("Director index, $\\log(1 + \\text{prior votes})$", "director_index"), ("Has star (share)", "has_star"),
    ("Runtime (minutes)", "runtime"), ("English-language original (share)", "is_english"),
    ("Release year", "release_year"), ("Number of votes", "vote_count"),
]
lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Descriptive statistics]{Descriptive statistics of the variables used in the models, for the "
         r"19,084 network films. Revenue and return on investment are available only for films that report them.}",
         r"\label{tab:descriptives}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrrr}", r"\hline",
         r"Variable & Films & Mean & SD & Min & Median & Max \\", r"\hline"]
fmt = lambda v: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
desc_rows = []
for label, col in VARS:
    if col is None:
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{label}}}}} \\")
        continue
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    st = [len(s), s.mean(), s.std(), s.min(), s.median(), s.max()]
    desc_rows.append({"variable": label, "column": col, "n": st[0], "mean": st[1], "sd": st[2], "min": st[3],
                      "median": st[4], "max": st[5]})
    f_ = (lambda v: f"{v:.0f}" if v == round(v) else f"{v:.1f}") if col == "release_year" else fmt
    lines.append(f"{label} & {st[0]:,} & " + " & ".join(f_(v) for v in st[1:]) + r" \\")
lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
(OUT / "r09_descriptive_statistics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
pd.DataFrame(desc_rows).round(5).to_csv(OUT / "r09_descriptive_statistics.csv", index=False)

# ── full regressions, M4 ─────────────────────────────────────────────────────────────────────────
NET = "z_novelty + z_novelty_sq + z_log_betweenness + z_participation + z_clustering"
CTRL = "C(release_year) + C(primary_genre_model) + z_runtime + is_english"
STAR = "z_star_index + z_director_index + has_star"
COMM = "C(community_group, Treatment('minor'))"
NEED = ["release_year", "primary_genre_model", "z_runtime", "is_english", "z_star_index", "z_director_index",
        "has_star", "community_group", "z_novelty", "z_novelty_sq", "z_log_betweenness", "z_participation", "z_clustering"]
OUTC = {"log_revenue": "Log revenue", "revenue_percentile_year": "Revenue pct.", "vote_average": "Rating",
        "log_popularity": "Log popularity"}
TERMS = [("z_novelty", "Novelty"), ("z_novelty_sq", "Novelty$^2$"), ("z_log_betweenness", "Betweenness (log)"),
         ("z_participation", "Participation"), ("z_clustering", "Clustering"), ("z_star_index", "Cast star index"),
         ("z_director_index", "Director index"), ("has_star", "Has star"), ("z_runtime", "Runtime"),
         ("is_english", "English-language original")]
fits = {}
for y in OUTC:
    d = df.dropna(subset=[y] + NEED)
    fits[y] = smf.ols(f"{y} ~ {CTRL} + {NET} + {STAR} + {COMM}", d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["community_group"])[0]})
published = {("revenue_percentile_year", "z_log_betweenness"): 0.012, ("revenue_percentile_year", "z_participation"): -0.008,
             ("log_revenue", "z_log_betweenness"): 0.093, ("vote_average", "has_star"): 0.174,
             ("log_popularity", "z_director_index"): 0.108}
for (y, t), v in published.items():
    assert round(fits[y].params[t], 3) == v, f"{y} {t}: {fits[y].params[t]:.4f} != published {v}"
star = lambda p: "^{***}" if p < .001 else "^{**}" if p < .01 else "^{*}" if p < .05 else ""
lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption[Full results of the full model M4]{Full results of model M4 for the four main outcomes. "
         r"Coefficients per standard deviation of continuous predictors, with standard errors clustered by "
         r"community in parentheses. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (not corrected for multiple testing).}",
         r"\label{tab:full_m4}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrr}", r"\hline",
         " & " + " & ".join(OUTC.values()) + r" \\", r"\hline"]
for t, label in TERMS:
    co = " & ".join(f"${fits[y].params[t]:.3f}{star(fits[y].pvalues[t])}$" for y in OUTC)
    se = " & ".join(f"({fits[y].bse[t]:.3f})" for y in OUTC)
    lines += [f"{label} & {co} \\\\", f" & {se} \\\\"]
lines += [r"\hline",
          "Release-year indicators & " + " & ".join(["Yes"] * 4) + r" \\",
          "Primary-genre indicators & " + " & ".join(["Yes"] * 4) + r" \\",
          "Community indicators (35 + pooled minor) & " + " & ".join(["Yes"] * 4) + r" \\",
          "Films & " + " & ".join(f"{int(fits[y].nobs):,}" for y in OUTC) + r" \\",
          "$R^2$ & " + " & ".join(f"{fits[y].rsquared:.3f}" for y in OUTC) + r" \\",
          "Adjusted $R^2$ & " + " & ".join(f"{fits[y].rsquared_adj:.3f}" for y in OUTC) + r" \\",
          r"\hline", r"\end{tabular}", r"\end{table}"]
(OUT / "r09_full_regressions_m4.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
pd.concat([pd.DataFrame({"outcome": y, "term": f.params.index, "coef": f.params.values, "se": f.bse.values,
                         "p_value": f.pvalues.values}) for y, f in fits.items()]).round(6).to_csv(
    OUT / "r09_full_regressions_m4.csv", index=False)
print("published coefficients reproduced ✓")
print((OUT / "r09_descriptive_statistics.tex").read_text())
print((OUT / "r09_full_regressions_m4.tex").read_text())
