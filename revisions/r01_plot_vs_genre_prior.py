"""Revision R1: does plot information add beyond genre when both compete on equal terms?

The published comparison (Notebook 06, cell 11; Table 5 of the thesis) adds a genre prior and a plot prior
to a baseline that already contains primary-genre fixed effects, so the genre prior can add little by
construction. This script
  1. reproduces the published numbers (same sample, folds and code), and
  2. adds a symmetric comparison on a baseline WITHOUT genre: year, runtime, English, star power.
Genre then enters in three forms (primary-genre dummies, all TMDB genre labels, genre prior) and plot
information in two (plot prior, full embedding), each on the same films and the same folds.

Run from the project root:  python revisions/r01_plot_vs_genre_prior.py
Writes outputs/tables/revisions/r01_plot_vs_genre_prior.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "processed" / "features"
OUT_DIR = ROOT / "outputs" / "tables" / "revisions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CFG = {"cv_folds": 5, "temporal_split_year": 2015, "prior_k": 20, "first_year": 1990, "window_years": 5,
       "ridge_alphas": [1, 3, 10, 30, 100, 300, 1000], "random_state": 42, "min_money_for_roi": 10_000}

# ── data exactly as Notebook 06 loads it ──────────────────────────────────────────────────────────
df = pd.read_csv(FEAT / "modeling_table_success_star_power.csv", low_memory=False)
E = np.load(FEAT / "sbert_minilm_embeddings_names_masked.npy").astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
emb_index = pd.read_csv(FEAT / "sbert_minilm_embedding_index_names_masked.csv")
emb_index["row"] = np.arange(len(E))
row_of = pd.Series(emb_index["row"].values, index=emb_index["tmdb_id"].values)
row_of = row_of[~row_of.index.duplicated()]
mm = CFG["min_money_for_roi"]
ok = (df["revenue_clean"] >= mm) & (df["budget_clean"] >= mm)
df["log_roi"] = np.where(ok, np.log(df["revenue_clean"].where(ok) / df["budget_clean"].where(ok)), np.nan)
df["primary_genre_model"] = df["primary_genre_model"].fillna("Other")
OUT = {"revenue_percentile_year": "Revenue percentile", "log_roi": "Log ROI",
       "vote_average": "Rating", "log_popularity": "Log popularity"}


def plot_prior(d, y, k):
    rows = d["tmdb_id"].map(row_of).values.astype(int)
    yrs, vals = d["release_year"].values, d[y].values
    out = np.full(len(d), np.nan)
    for yr in np.unique(yrs):
        q, pool = np.where(yrs == yr)[0], np.where(yrs < yr)[0]
        if len(pool) < k:
            continue
        S = E[rows[q]] @ E[rows[pool]].T
        top = np.argpartition(S, -k, axis=1)[:, -k:]
        w = np.clip(np.take_along_axis(S, top, axis=1), 0, None)
        out[q] = (w * vals[pool][top]).sum(1) / np.maximum(w.sum(1), 1e-9)
    return out


def genre_prior(d, y):
    g = d.groupby(["primary_genre_model", "release_year"])[y].agg(["sum", "count"]).sort_index()
    cs = g.groupby(level=0).cumsum() - g
    prior = (cs["sum"] / cs["count"].replace(0, np.nan)).rename("genre_prior")
    return d.join(prior, on=["primary_genre_model", "release_year"])["genre_prior"].values


base_cols = ["z_runtime", "is_english", "z_star_index", "z_director_index", "has_star"]
kf = KFold(CFG["cv_folds"], shuffle=True, random_state=CFG["random_state"])
rows = []
for y, yname in OUT.items():
    d = df.dropna(subset=[y] + base_cols + ["primary_genre_model", "release_year"]).copy()
    d = d.sort_values("release_year").reset_index(drop=True)
    d["plot_prior"] = plot_prior(d, y, CFG["prior_k"])
    d["genre_prior"] = genre_prior(d, y)
    d = d[d["release_year"] >= CFG["first_year"] + CFG["window_years"]]
    d = d.dropna(subset=["plot_prior", "genre_prior"]).reset_index(drop=True)
    yv = d[y].values
    emb = E[d["tmdb_id"].map(row_of).values.astype(int)]

    yr_d = pd.get_dummies(d["release_year"].astype(str), prefix="yr", drop_first=True).astype(float)
    pg_d = pd.get_dummies(d["primary_genre_model"].astype(str), prefix="pg", drop_first=True).astype(float)
    all_g = d["genres_clean"].fillna("").str.get_dummies(sep=", ").add_prefix("g_").astype(float)
    base = d[base_cols].astype(float).reset_index(drop=True)
    gp, pp = d[["genre_prior"]].astype(float), d[["plot_prior"]].astype(float)

    def X(*parts, year="dummies"):
        yp = yr_d if year == "dummies" else pd.DataFrame({"year": d["release_year"].values - 2000.0})
        return pd.concat([yp, base, *parts], axis=1).values

    specs = {  # name: (comparison, parts, uses_embedding)
        "published: baseline (with primary genre)": ("published", [pg_d], False),
        "published: + genre prior": ("published", [pg_d, gp], False),
        "published: + plot prior": ("published", [pg_d, pp], False),
        "published: + full embedding": ("published", [pg_d], True),
        "no-genre baseline": ("symmetric", [], False),
        "+ primary-genre dummies": ("symmetric", [pg_d], False),
        "+ all genre labels": ("symmetric", [all_g], False),
        "+ genre prior": ("symmetric", [gp], False),
        "+ plot prior": ("symmetric", [pp], False),
        "+ full embedding": ("symmetric", [], True),
        "+ all genre labels + plot prior": ("symmetric", [all_g, pp], False),
        "+ all genre labels + full embedding": ("symmetric", [all_g], True),
    }
    tr_t = np.where(d["release_year"].values < CFG["temporal_split_year"])[0]
    te_t = np.where(d["release_year"].values >= CFG["temporal_split_year"])[0]
    for name, (comp, parts, use_emb) in specs.items():
        Xc, Xt = X(*parts), X(*parts, year="linear")
        if use_emb:
            Xc, Xt = np.hstack([Xc, emb]), np.hstack([Xt, emb])
        mk = (lambda: RidgeCV(alphas=CFG["ridge_alphas"])) if use_emb else LinearRegression
        scores = [r2_score(yv[te], mk().fit(Xc[tr], yv[tr]).predict(Xc[te])) for tr, te in kf.split(d)]
        fwd = r2_score(yv[te_t], mk().fit(Xt[tr_t], yv[tr_t]).predict(Xt[te_t]))
        rows.append({"outcome": yname, "comparison": comp, "features": name, "n_films": len(d),
                     "cv_r2": np.mean(scores), "cv_r2_sd": np.std(scores), "forward_r2": fwd})
    print(f"{yname:20s} n={len(d):,}")

res = pd.DataFrame(rows)
first = res.groupby(["outcome", "comparison"])
res["cv_gain"] = res["cv_r2"] - first["cv_r2"].transform("first")
res["forward_gain"] = res["forward_r2"] - first["forward_r2"].transform("first")
res.round(5).to_csv(OUT_DIR / "r01_plot_vs_genre_prior.csv", index=False)
pd.set_option("display.width", 200)
for comp in ["published", "symmetric"]:
    print(f"\n=== {comp} ===")
    print(res[res["comparison"] == comp].pivot(index="features", columns="outcome", values="cv_gain")
          .reindex(res[res["comparison"] == comp]["features"].unique()).round(4))
    print(res[res["comparison"] == comp].pivot(index="features", columns="outcome", values="forward_gain")
          .reindex(res[res["comparison"] == comp]["features"].unique()).round(4).add_prefix("fwd "))
print("\nBaseline R²:\n", res[res["features"].isin(["published: baseline (with primary genre)", "no-genre baseline"])]
      [["outcome", "features", "n_films", "cv_r2", "forward_r2"]].round(4).to_string(index=False))
