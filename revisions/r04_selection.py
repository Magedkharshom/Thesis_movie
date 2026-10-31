"""Revision R4: selection from the 25-vote threshold and from keeping only the largest connected component.

  A. Who is excluded? The 835 network-corpus films outside the largest component are compared with the
     19,084 network films: novelty (same definition as Notebook 05: mean cosine distance to the 20 most
     similar overviews among all 19,919), number of mutual neighbours, votes, rating, revenue, language, year.
  B. Does the exclusion hide a sweet spot of novelty? The novelty models (controls, star power, novelty and
     novelty²; HC3 errors) are estimated on the network films and on all 19,919 films, with star power and the
     revenue percentile rebuilt for every film exactly as in Notebook 05 (checked against the modelling table).
  C. Does the vote threshold drive the results? The full model M4 is re-estimated on films with at least
     25 (all), 50, 100, and 250 votes; if selection on visibility created the network associations, they
     should change systematically as the threshold rises.
  D. What does the threshold remove? Thesis-corpus films with an overview of at least 35 words, with and
     without 25 votes, are compared.

Run from the project root:  python revisions/r04_selection.py
Writes outputs/tables/revisions/r04_*.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
FEAT, CORE = ROOT / "processed" / "features", ROOT / "processed" / "core"
OUT_DIR = ROOT / "outputs" / "tables" / "revisions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
K, CAST_TOP_N, STAR_TOP, MIN_GENRE = 20, 5, 0.10, 100
pd.set_option("display.width", 220)

model = pd.read_csv(FEAT / "modeling_table_success_star_power.csv", low_memory=False)
thesis = pd.read_csv(CORE / "df_thesis_films.csv", low_memory=False,
                     usecols=["tmdb_id", "original_language", "release_year", "primary_genre", "overview_word_count",
                              "runtime", "cast_clean", "director_name", "revenue_clean", "budget_clean",
                              "popularity", "vote_average", "vote_count"])
netc = pd.read_csv(CORE / "df_network_corpus.csv", usecols=["tmdb_id", "title"]).drop_duplicates("tmdb_id")
print(f"thesis corpus {len(thesis):,} · network corpus {len(netc):,} · network (LCC) {len(model):,}")

# ── embeddings, novelty and mutual k-NN for all 19,919 films ─────────────────────────────────────────
E = np.load(FEAT / "sbert_minilm_embeddings_names_masked.npy").astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
idx = pd.read_csv(FEAT / "sbert_minilm_embedding_index_names_masked.csv")
n = len(E)
nbrs, sims = np.empty((n, K), dtype=np.int32), np.empty((n, K), dtype=np.float32)
for s in range(0, n, 2000):
    S = E[s:s + 2000] @ E.T
    S[np.arange(len(S)), np.arange(s, s + len(S))] = -np.inf
    top = np.argpartition(S, -K, axis=1)[:, -K:]
    nbrs[s:s + len(S)], sims[s:s + len(S)] = top, np.take_along_axis(S, top, axis=1)
novelty = 1 - sims.mean(1)
nbr_sets = [set(r) for r in nbrs]
mutual_deg = np.array([sum(i in nbr_sets[j] for j in nbrs[i]) for i in range(n)])
emb = pd.DataFrame({"tmdb_id": idx["tmdb_id"].values, "novelty_all": novelty, "mutual_neighbours": mutual_deg})
emb = emb.drop_duplicates("tmdb_id")

d = netc.merge(emb, on="tmdb_id", how="left").merge(thesis, on="tmdb_id", how="left")
d["in_network"] = d["tmdb_id"].isin(model["tmdb_id"])
chk = d[d["in_network"]].merge(model[["tmdb_id", "novelty"]], on="tmdb_id")
print(f"novelty matches Notebook 05: max abs diff {np.abs(chk['novelty'] - chk['novelty_all']).max():.2e}")

# ── A. who is excluded ───────────────────────────────────────────────────────────────────────────────
d["revenue_known"] = d["revenue_clean"].notna() & (d["revenue_clean"] > 0)
d["english"] = d["original_language"].eq("en")
top10, top1 = d["novelty_all"].quantile(0.90), d["novelty_all"].quantile(0.99)
grp = d.groupby("in_network")
A = pd.DataFrame({
    "films": grp.size(),
    "median_novelty": grp["novelty_all"].median(),
    "share_top10pct_novelty": grp["novelty_all"].apply(lambda x: (x >= top10).mean()),
    "share_top1pct_novelty": grp["novelty_all"].apply(lambda x: (x >= top1).mean()),
    "share_no_mutual_neighbour": grp["mutual_neighbours"].apply(lambda x: (x == 0).mean()),
    "median_votes": grp["vote_count"].median(), "median_rating": grp["vote_average"].median(),
    "median_popularity": grp["popularity"].median(), "share_revenue_known": grp["revenue_known"].mean(),
    "share_english": grp["english"].mean(), "median_year": grp["release_year"].median(),
}).rename(index={True: "network (LCC)", False: "excluded"}).T
A.round(4).to_csv(OUT_DIR / "r04_excluded_films.csv")
print("\nA. Excluded films vs network films:\n", A.round(3))
ex = d[~d["in_network"]]
print(f"Excluded films among the 1% most novel overviews: {(ex['novelty_all'] >= top1).sum()} of "
      f"{(d['novelty_all'] >= top1).sum()} ({(ex['novelty_all'] >= top1).sum() / (d['novelty_all'] >= top1).sum():.0%})")

# ── rebuild star power and revenue percentile for every film, as in Notebook 05 ─────────────────────
def prior_career(pf):
    py = (pf.groupby(["person", "release_year"]).agg(n=("tmdb_id", "size"), v=("votes", "sum")).reset_index()
          .sort_values(["person", "release_year"]))
    g = py.groupby("person")
    py["prior_films"], py["prior_votes"] = g["n"].cumsum() - py["n"], g["v"].cumsum() - py["v"]
    return pf.merge(py[["person", "release_year", "prior_films", "prior_votes"]], on=["person", "release_year"])

base = thesis[["tmdb_id", "release_year", "vote_count", "cast_clean", "director_name"]].copy()
base["votes"] = pd.to_numeric(base["vote_count"], errors="coerce").fillna(0)
cast = base.assign(person=base["cast_clean"].fillna("").str.split(r"\s*,\s*", regex=True)).explode("person")
cast["person"] = cast["person"].str.strip()
cast = cast[cast["person"].ne("") & cast["person"].notna()].groupby("tmdb_id").head(CAST_TOP_N)
cast = prior_career(cast[["person", "tmdb_id", "release_year", "votes"]])
star = cast.groupby("tmdb_id").agg(star_prior_votes_max=("prior_votes", "max")).reset_index()
dirs = base.assign(person=base["director_name"].fillna("").str.split(r"\s*,\s*", regex=True).str[0].str.strip())
dirs = prior_career(dirs[dirs["person"].ne("")][["person", "tmdb_id", "release_year", "votes"]])
star = star.merge(dirs.groupby("tmdb_id").agg(director_prior_votes=("prior_votes", "max")).reset_index(),
                  on="tmdb_id", how="outer")
rev = thesis[thesis["revenue_clean"].notna() & (thesis["revenue_clean"] > 0)][["tmdb_id", "release_year", "revenue_clean"]]
rev = rev.assign(rp=rev.groupby("release_year")["revenue_clean"].rank(pct=True))[["tmdb_id", "rp"]]
d = d.merge(star, on="tmdb_id", how="left").merge(rev, on="tmdb_id", how="left")
d[["star_prior_votes_max", "director_prior_votes"]] = d[["star_prior_votes_max", "director_prior_votes"]].fillna(0)
d["star_index"], d["director_index"] = np.log1p(d["star_prior_votes_max"]), np.log1p(d["director_prior_votes"])
c = d.merge(model[["tmdb_id", "star_index", "director_index", "revenue_percentile_year"]], on="tmdb_id", suffixes=("", "_m"))
print(f"\nrebuilt vs Notebook 05 (network films): star index r={c['star_index'].corr(c['star_index_m']):.4f}, "
      f"director index r={c['director_index'].corr(c['director_index_m']):.4f}, "
      f"revenue percentile r={c['rp'].corr(c['revenue_percentile_year']):.4f}")

# ── B. novelty models with and without the excluded films ────────────────────────────────────────────
d["log_popularity"] = np.log1p(d["popularity"])
d["log_revenue"] = np.log(d["revenue_clean"].where(d["revenue_clean"] > 0))
gc = d["primary_genre"].fillna("Unknown")
d["genre"] = gc.where(gc.map(gc.value_counts()) >= MIN_GENRE, "Other")
d["is_english"] = d["english"].astype(int)
d["has_star"] = (d["star_index"] >= d.groupby("release_year")["star_index"].transform(lambda x: x.quantile(1 - STAR_TOP))).astype(int)
zs = lambda s, ref: (s - ref.mean()) / ref.std()
for col in ["novelty_all", "star_index", "director_index", "runtime"]:
    d[f"z_{col}"] = zs(d[col], d.loc[d["in_network"], col])          # standardised on the network films, as published
d["z_nov_sq"] = d["z_novelty_all"] ** 2
F = ("C(release_year) + C(genre) + z_runtime + is_english + z_star_index + z_director_index + has_star"
     " + z_novelty_all + z_nov_sq")
B_rows = []
for y, yname in {"rp": "Revenue percentile", "log_revenue": "Log revenue", "vote_average": "Rating",
                 "log_popularity": "Log popularity"}.items():
    for sample, data in {"network films": d[d["in_network"]], "all network-corpus films": d}.items():
        data = data.dropna(subset=[y, "z_runtime"])
        r = smf.ols(f"{y} ~ {F}", data).fit(cov_type="HC3")
        b1, b2 = r.params["z_novelty_all"], r.params["z_nov_sq"]
        lo, hi = data["z_novelty_all"].quantile([0.01, 0.99])
        B_rows.append({"outcome": yname, "sample": sample, "n": int(r.nobs),
                       "excluded_in_sample": int((~data["in_network"]).sum()),
                       "novelty": b1, "p_novelty": r.pvalues["z_novelty_all"], "novelty_sq": b2,
                       "p_novelty_sq": r.pvalues["z_nov_sq"],
                       "vertex_sd": -b1 / (2 * b2) if b2 < 0 else np.nan,
                       "range_sd_p1_p99": f"{lo:.2f} to {hi:.2f}"})
B = pd.DataFrame(B_rows)
B.round(5).to_csv(OUT_DIR / "r04_novelty_with_excluded.csv", index=False)
print("\nB. Novelty and novelty² with and without the excluded films (HC3):\n", B.round(4).to_string(index=False))

# ── C. M4 on rising vote thresholds ──────────────────────────────────────────────────────────────────
NET = "z_novelty + z_novelty_sq + z_log_betweenness + z_participation + z_clustering"
CTRL = "C(release_year) + C(primary_genre_model) + z_runtime + is_english"
STAR = "z_star_index + z_director_index + has_star"
COMM = "C(community_group, Treatment('minor'))"
TERMS = ["z_novelty", "z_novelty_sq", "z_log_betweenness", "z_participation", "z_clustering"]
C_rows = []
for y, yname in {"revenue_percentile_year": "Revenue percentile", "vote_average": "Rating",
                 "log_popularity": "Log popularity"}.items():
    for thr in [25, 50, 100, 250]:
        data = model[model["vote_count"] >= thr].dropna(subset=[y, "z_runtime", "z_star_index"] + TERMS)
        fit = lambda f: smf.ols(f, data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data["community_group"])[0]})
        m3n, m4 = fit(f"{y} ~ {CTRL} + {STAR} + {COMM}"), fit(f"{y} ~ {CTRL} + {NET} + {STAR} + {COMM}")
        row = {"outcome": yname, "min_votes": thr, "n": int(m4.nobs), "r2_gain_network": m4.rsquared - m3n.rsquared}
        for t in TERMS[2:]:
            row[t], row[f"p_{t}"] = m4.params[t], m4.pvalues[t]
        C_rows.append(row)
C = pd.DataFrame(C_rows)
C.round(5).to_csv(OUT_DIR / "r04_vote_thresholds.csv", index=False)
print("\nC. Full model M4 on rising vote thresholds:\n", C.round(4).to_string(index=False))

# ── D. what the 25-vote threshold removes ────────────────────────────────────────────────────────────
t = thesis[thesis["overview_word_count"] >= 35].copy()
t["kept"] = np.where(t["vote_count"] >= 25, "25+ votes", "1-24 votes")
t["revenue_known"] = t["revenue_clean"].notna() & (t["revenue_clean"] > 0)
t["english"] = t["original_language"].eq("en")
t = t.merge(rev, on="tmdb_id", how="left")
g = t.groupby("kept")
D = pd.DataFrame({"films": g.size(), "share_revenue_known": g["revenue_known"].mean(),
                  "median_revenue_pct_if_known": g["rp"].median(), "share_english": g["english"].mean(),
                  "median_year": g["release_year"].median(), "median_rating": g["vote_average"].median(),
                  "median_votes": g["vote_count"].median()}).T
D.round(4).to_csv(OUT_DIR / "r04_vote_threshold_profile.csv")
print("\nD. Thesis-corpus films with 35+ word overviews, by vote threshold:\n", D.round(3))
print(f"Share of all revenue-reporting thesis films that pass the threshold: "
      f"{t.loc[t['revenue_known'], 'kept'].eq('25+ votes').mean():.1%}")
