"""Revision R2: are the national-cinema communities driven by stories, or by place names and language?

Four checks on the existing network, without re-embedding:
  A. language composition: NMI between community and original language, and the share of each
     national community's films in its expected language(s);
  B. explicit geographic cues: share of (person-masked) overviews in which spaCy finds a place,
     nationality or language (GPE, LOC, NORP, LANGUAGE);
  C. films with NO such cue: are they still firmly placed in their national community
     (membership confidence, share of their edge weight inside the community)?
  D. a representation without the cues: TF-IDF vectors of the overviews with person names masked
     AND all GPE/LOC/NORP/LANGUAGE spans removed. For films of each community, the share of their
     20 nearest TF-IDF neighbours in the same community, against the share expected by chance.
     The same is reported for TF-IDF with the cues kept and for the SBERT embeddings.

Run from the project root:  python revisions/r02_national_communities.py
Writes outputs/tables/revisions/r02_*.csv
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score as nmi

ROOT = Path(__file__).resolve().parents[1]
FEAT, NET = ROOT / "processed" / "features", ROOT / "processed" / "network"
OUT_DIR = ROOT / "outputs" / "tables" / "revisions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CUES = {"GPE", "LOC", "NORP", "LANGUAGE"}
K = 20
NATIONAL = {7: ["fr"], 8: ["ja", "ko"], 10: ["es", "pt"], 15: ["it"], 19: ["hi", "ta", "te", "ml", "kn", "bn", "mr"],
            28: ["da", "sv", "no", "nb", "fi", "is"], 33: ["tr", "fa", "ar", "ku"]}

df = pd.read_csv(FEAT / "modeling_table_success_star_power.csv", low_memory=False,
                 usecols=["tmdb_id", "node_id", "title", "community_id", "community_group", "original_language",
                          "primary_genre_model", "membership_confidence"])
labels = pd.read_csv(ROOT / "outputs" / "tables" / "community_manual_labeling_template_labeled.csv")
MAJOR = labels["community_id"].tolist()
LAB = labels.set_index("community_id")["suggested_label_manual"]
masked = pd.read_csv(FEAT / "overview_person_names_masked.csv").drop_duplicates("tmdb_id")
df = df.merge(masked[["tmdb_id", "overview_masked"]], on="tmdb_id", how="left")
assert df["overview_masked"].notna().all(), "every network film needs a masked overview"
df["comm"] = np.where(df["community_id"].isin(MAJOR), df["community_id"], -1)   # -1 = pooled minor

# ── A. language composition ────────────────────────────────────────────────────────────────────
lang = df["original_language"].fillna("xx")
lang_top = lang.where(lang.map(lang.value_counts()) >= 50, "other")
shuffled = np.random.default_rng(42).permutation(lang_top.values)
print(f"NMI(community, original language) = {nmi(df['comm'], lang_top):.3f}  "
      f"(shuffled {nmi(df['comm'], shuffled):.3f}; community vs primary genre "
      f"{nmi(df['comm'], df['primary_genre_model'].fillna('Other')):.3f})")
print(f"NMI(community, English vs not)    = {nmi(df['comm'], lang.eq('en')):.3f}")

# ── B. explicit geographic cues (spaCy NER on the person-masked overviews) ─────────────────────
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
has_cue, stripped = [], []
for doc in nlp.pipe(df["overview_masked"].tolist(), batch_size=256):
    spans = [e for e in doc.ents if e.label_ in CUES]
    has_cue.append(len(spans) > 0)
    text, last = [], 0
    for e in spans:
        text.append(doc.text[last:e.start_char]); last = e.end_char
    text.append(doc.text[last:])
    stripped.append(" ".join(text))
df["has_geo_cue"] = has_cue
df["overview_no_cues"] = stripped
# stricter variant: also drop every capitalised word that does not start a sentence, which removes
# person names the recognizer missed (e.g. Antoine, Massimo, Priya) together with remaining proper nouns
_cap = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+\b")
df["overview_no_proper"] = [_cap.sub(" ", t) for t in stripped]

# ── C. firmness of films without a cue ───────────────────────────────────────────────────────────
edges = pd.read_csv(NET / "semantic_network_edges_mutual_knn.csv",
                    usecols=["source_node_id", "target_node_id", "cosine_similarity"])
c_of = df.set_index("node_id")["comm"]
e = edges.assign(cs=edges["source_node_id"].map(c_of), ct=edges["target_node_id"].map(c_of)).dropna()
long = pd.concat([e.rename(columns={"source_node_id": "n", "cs": "c", "ct": "o"})[["n", "c", "o", "cosine_similarity"]],
                  e.rename(columns={"target_node_id": "n", "ct": "c", "cs": "o"})[["n", "c", "o", "cosine_similarity"]]])
long["inside"] = (long["c"] == long["o"]) * long["cosine_similarity"]
inside = long.groupby("n")[["inside", "cosine_similarity"]].sum()
df["inside_share"] = df["node_id"].map(inside["inside"] / inside["cosine_similarity"])

# ── D. nearest-neighbour community agreement under three representations ───────────────────────
def same_comm_share(X, comm):
    """Share of each film's K nearest neighbours (cosine, self excluded) in the same community."""
    out = np.empty(X.shape[0])
    for s in range(0, X.shape[0], 2000):
        S = X[s:s + 2000] @ X.T
        S = S.toarray() if hasattr(S, "toarray") else S
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        top = np.argpartition(S, -K, axis=1)[:, -K:]
        out[s:s + S.shape[0]] = (comm[top] == comm[s:s + S.shape[0], None]).mean(1)
    return out

comm = df["comm"].values
tf_kw = dict(stop_words="english", min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
X_cues = TfidfVectorizer(**tf_kw).fit_transform(df["overview_masked"].str.replace("someone", " ", regex=False))
X_none = TfidfVectorizer(**tf_kw).fit_transform(df["overview_no_cues"].str.replace("someone", " ", regex=False))
X_noprop = TfidfVectorizer(**tf_kw).fit_transform(df["overview_no_proper"].str.replace("someone", " ", regex=False))
E = np.load(FEAT / "sbert_minilm_embeddings_names_masked.npy").astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
idx = pd.read_csv(FEAT / "sbert_minilm_embedding_index_names_masked.csv")
row = pd.Series(np.arange(len(idx)), index=idx["tmdb_id"])
row = row[~row.index.duplicated()]
E = E[df["tmdb_id"].map(row).values.astype(int)]
df["knn_same_sbert"] = same_comm_share(E, comm)
df["knn_same_tfidf_with_cues"] = same_comm_share(X_cues, comm)
df["knn_same_tfidf_no_cues"] = same_comm_share(X_none, comm)
df["knn_same_tfidf_no_proper"] = same_comm_share(X_noprop, comm)

# ── tables ─────────────────────────────────────────────────────────────────────────────────────
n = len(df)
rows = []
for c in MAJOR:
    g = df[df["comm"] == c]
    exp_lang = NATIONAL.get(c)
    rows.append({
        "community_id": c, "label": LAB[c], "national": c in NATIONAL, "films": len(g),
        "top_language": g["original_language"].value_counts().index[0],
        "top_language_share": g["original_language"].value_counts(normalize=True).iloc[0],
        "expected_language_share": g["original_language"].isin(exp_lang).mean() if exp_lang else np.nan,
        "english_share": g["original_language"].eq("en").mean(),
        "geo_cue_share": g["has_geo_cue"].mean(),
        "confidence_with_cue": g.loc[g["has_geo_cue"], "membership_confidence"].mean(),
        "confidence_no_cue": g.loc[~g["has_geo_cue"], "membership_confidence"].mean(),
        "inside_share_with_cue": g.loc[g["has_geo_cue"], "inside_share"].mean(),
        "inside_share_no_cue": g.loc[~g["has_geo_cue"], "inside_share"].mean(),
        "chance_share": len(g) / n,
        "knn_same_sbert": g["knn_same_sbert"].mean(),
        "knn_same_tfidf_with_cues": g["knn_same_tfidf_with_cues"].mean(),
        "knn_same_tfidf_no_cues": g["knn_same_tfidf_no_cues"].mean(),
        "knn_same_tfidf_no_proper": g["knn_same_tfidf_no_proper"].mean(),
    })
tab = pd.DataFrame(rows)
tab["lift_tfidf_no_cues"] = tab["knn_same_tfidf_no_cues"] / tab["chance_share"]
tab["lift_tfidf_no_proper"] = tab["knn_same_tfidf_no_proper"] / tab["chance_share"]
tab.round(4).to_csv(OUT_DIR / "r02_national_communities.csv", index=False)

grp = tab.assign(kind=np.where(tab["community_id"].eq(28), "C28 (mixed)",
                              np.where(tab["national"], "cultural (6)", "other major"))).groupby("kind")
w = lambda col: grp.apply(lambda g: np.average(g[col], weights=g["films"]), include_groups=False)
summary = pd.DataFrame({c: w(c) for c in ["geo_cue_share", "confidence_with_cue", "confidence_no_cue",
                                          "inside_share_with_cue", "inside_share_no_cue", "chance_share",
                                          "knn_same_sbert", "knn_same_tfidf_with_cues", "knn_same_tfidf_no_cues",
                                          "knn_same_tfidf_no_proper"]})
summary.round(4).to_csv(OUT_DIR / "r02_summary.csv")
pd.set_option("display.width", 250)
print(f"\nOverviews with a place/nationality/language cue: {df['has_geo_cue'].mean():.1%} of network films")
print("\nFilm-weighted means:\n", summary.round(3).T)
cols = ["community_id", "label", "films", "top_language", "top_language_share", "expected_language_share",
        "geo_cue_share", "confidence_with_cue", "confidence_no_cue", "inside_share_no_cue",
        "chance_share", "knn_same_sbert", "knn_same_tfidf_with_cues", "knn_same_tfidf_no_cues", "lift_tfidf_no_cues", "lift_tfidf_no_proper"]
print("\nNational communities:\n", tab[tab["national"]][cols].round(3).to_string(index=False))
