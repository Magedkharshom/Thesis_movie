"""Revision R3: does the partial look-ahead in the star-power measures change the results?

The cast star index and director index count only earlier films, but they weight them by the TMDB votes those
films hold in the data extract (collected around May 2026), which include votes cast after the focal film's
release. This script re-estimates the nested models M1–M4 of Notebook 05 with star-power measures that cannot
look ahead: the NUMBER of earlier films of the cast member with the longest career and of the director
(log(1 + count)), and "has star" defined as the top 10% of that experience within the release year.
Everything else (sample, controls, network terms, community clustering) is as in Notebook 05.

Run from the project root:  python revisions/r03_star_power_without_vote_lookahead.py
Writes outputs/tables/revisions/r03_*.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "tables" / "revisions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "processed" / "features" / "modeling_table_success_star_power.csv", low_memory=False)

z = lambda s: (s - s.mean()) / s.std()
df["z_star_exp"] = z(np.log1p(df["star_prior_films_max"]))
df["z_director_exp"] = z(np.log1p(df["director_prior_films"]))
q90 = df.groupby("release_year")["star_prior_films_max"].transform(lambda x: x.quantile(0.9))
df["has_star_exp"] = (df["star_prior_films_max"] >= q90).astype(int)

NET = "z_novelty + z_novelty_sq + z_log_betweenness + z_participation + z_clustering"
CTRL = "C(release_year) + C(primary_genre_model) + z_runtime + is_english"
COMM = "C(community_group, Treatment('minor'))"
STARS = {"published (prior votes)": "z_star_index + z_director_index + has_star",
         "experience (prior film counts)": "z_star_exp + z_director_exp + has_star_exp"}
OUTCOMES = {"log_revenue": "Log revenue", "revenue_percentile_year": "Revenue percentile",
            "vote_average": "Rating", "log_popularity": "Log popularity"}
NEED = ["release_year", "primary_genre_model", "z_runtime", "is_english", "z_star_index", "z_director_index",
        "has_star", "community_group", "z_novelty", "z_novelty_sq", "z_log_betweenness", "z_participation",
        "z_clustering", "z_star_exp", "z_director_exp", "has_star_exp"]
TERMS = ["z_novelty", "z_novelty_sq", "z_log_betweenness", "z_participation", "z_clustering"]


def fit(formula, d):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["community_group"])[0]})


fit_rows, coef_rows = [], []
for y, yname in OUTCOMES.items():
    d = df.dropna(subset=[y] + NEED).copy()
    m1 = fit(f"{y} ~ {CTRL}", d)
    m2 = fit(f"{y} ~ {CTRL} + {NET}", d)
    for sname, star in STARS.items():
        m3 = fit(f"{y} ~ {CTRL} + {NET} + {star}", d)
        m4 = fit(f"{y} ~ {CTRL} + {NET} + {star} + {COMM}", d)
        w = m4.wald_test(", ".join(f"{t} = 0" for t in TERMS), scalar=True)
        fit_rows.append({"outcome": yname, "star_measure": sname, "n": int(m4.nobs), "r2_M1": m1.rsquared,
                         "r2_M2": m2.rsquared, "r2_M3": m3.rsquared, "r2_M4": m4.rsquared,
                         "gain_network": m2.rsquared - m1.rsquared, "gain_star": m3.rsquared - m2.rsquared,
                         "gain_community": m4.rsquared - m3.rsquared, "network_block_p": float(w.pvalue)})
        ci = m4.conf_int()
        for t in TERMS + star.split(" + "):
            coef_rows.append({"outcome": yname, "star_measure": sname, "term": t, "coef": m4.params[t],
                              "ci_low": ci.loc[t, 0], "ci_high": ci.loc[t, 1], "p_value": m4.pvalues[t]})
    print(f"{yname:20s} n={int(m4.nobs):,}")

fits, coefs = pd.DataFrame(fit_rows), pd.DataFrame(coef_rows)
fits.round(5).to_csv(OUT_DIR / "r03_model_fit.csv", index=False)
coefs.round(5).to_csv(OUT_DIR / "r03_coefficients.csv", index=False)
pd.set_option("display.width", 220)
print("\nR² and block gains:\n", fits.round(4).to_string(index=False))
c = coefs[coefs["term"].isin(TERMS)].copy()
c["cell"] = c["coef"].map("{:+.4f}".format) + c["p_value"].map(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
print("\nNetwork coefficients in M4:\n", c.pivot(index=["outcome", "term"], columns="star_measure", values="cell").to_string())
s = coefs[~coefs["term"].isin(TERMS)].copy()
print("\nStar terms in M4:\n", s.round(4).to_string(index=False))
print("\nCorrelation of star measures (Spearman):\n",
      df[["z_star_index", "z_star_exp", "z_director_index", "z_director_exp"]].corr(method="spearman").round(2))
