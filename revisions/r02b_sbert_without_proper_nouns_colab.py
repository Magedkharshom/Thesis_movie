"""Revision R2b (optional, run in Colab): the R2 test with the sentence-embedding model itself.

R2 tests the cultural communities with TF-IDF, because the embedding model could not be downloaded in the
environment where R2 was run. This script repeats the key comparison with all-MiniLM-L6-v2: it re-embeds
the person-masked overviews after removing place, nationality and language expressions and every
capitalised word that does not start a sentence, and reports, for each major community, the share of its
films' 20 nearest neighbours that fall in the same community, with the original and the stripped text.

In Colab:  !pip -q install spacy sentence-transformers && python -m spacy download en_core_web_sm
           %run /content/drive/MyDrive/thesis_movie_project/revisions/r02b_sbert_without_proper_nouns_colab.py
Writes outputs/tables/revisions/r02b_sbert_without_proper_nouns.csv (about 5 minutes on a GPU).
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer

ROOT = Path("/content/drive/MyDrive/thesis_movie_project")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "processed" / "features"
OUT = ROOT / "outputs" / "tables" / "revisions"
OUT.mkdir(parents=True, exist_ok=True)
K, CUES = 20, {"GPE", "LOC", "NORP", "LANGUAGE"}
CULTURAL = [7, 8, 10, 15, 19, 33]

df = pd.read_csv(FEAT / "modeling_table_success_star_power.csv", low_memory=False,
                 usecols=["tmdb_id", "community_id"])
labels = pd.read_csv(ROOT / "outputs" / "tables" / "community_manual_labeling_template_labeled.csv")
major = labels["community_id"].tolist()
masked = pd.read_csv(FEAT / "overview_person_names_masked.csv").drop_duplicates("tmdb_id")
df = df.merge(masked[["tmdb_id", "overview_masked"]], on="tmdb_id")
comm = np.where(df["community_id"].isin(major), df["community_id"], -1)

nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
cap = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+\b")
stripped = []
for doc in nlp.pipe(df["overview_masked"].tolist(), batch_size=256):
    text, last = [], 0
    for e in doc.ents:
        if e.label_ in CUES:
            text.append(doc.text[last:e.start_char]); last = e.end_char
    text.append(doc.text[last:])
    stripped.append(cap.sub(" ", " ".join(text)))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
enc = lambda texts: model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
E_orig, E_strip = enc(df["overview_masked"].tolist()), enc(stripped)


def same_comm_share(X):
    out = np.empty(len(X))
    for s in range(0, len(X), 2000):
        S = X[s:s + 2000] @ X.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -np.inf
        top = np.argpartition(S, -K, axis=1)[:, -K:]
        out[s:s + len(S)] = (comm[top] == comm[s:s + len(S), None]).mean(1)
    return out


df["same_orig"], df["same_stripped"] = same_comm_share(E_orig), same_comm_share(E_strip)
tab = (df[df["community_id"].isin(major)].groupby("community_id")
       .agg(films=("tmdb_id", "size"), same_orig=("same_orig", "mean"), same_stripped=("same_stripped", "mean"))
       .reset_index())
tab["chance"] = tab["films"] / len(df)
tab["lift_orig"], tab["lift_stripped"] = tab["same_orig"] / tab["chance"], tab["same_stripped"] / tab["chance"]
tab["cultural"] = tab["community_id"].isin(CULTURAL)
tab.round(4).to_csv(OUT / "r02b_sbert_without_proper_nouns.csv", index=False)
for flag, g in tab.groupby("cultural"):
    w = g["films"] / g["films"].sum()
    print(f"{'cultural (6)' if flag else 'other major':13s} same-community neighbours: original "
          f"{(w * g['same_orig']).sum():.3f}, stripped {(w * g['same_stripped']).sum():.3f}; "
          f"lift stripped {g['lift_stripped'].min():.1f}–{g['lift_stripped'].max():.1f}")
print(tab[tab["cultural"]].round(3).to_string(index=False))
