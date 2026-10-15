# Narrative similarity and film success — thesis project

This project builds a **plot-similarity network** of 19,084 feature films (1990–2024). Each film's plot overview is embedded with sentence-BERT, and two films are linked when their plots are mutual nearest neighbours. It then asks whether a film's **place in that network** — which narrative community it belongs to, how typical or bridging its story is, and *when* the story was told — is associated with commercial success, audience ratings and attention.

All success results are **associations, not causal effects**.

---

## Pipeline

Run the notebooks in order in Google Colab. Each one reads only what the earlier ones wrote.

| # | Notebook | What it does | Main outputs |
|---|---|---|---|
| 01 | `01_build_curated_thesis_corpus_v2.ipynb` | Filters the raw TMDB extract (1.19 M titles) to the thesis corpus (98,037 films), the semantic corpus (20,091) and the network corpus (19,919). Removes recordings without a plot. | `processed/core/` |
| 02 | `02_eda_semantic_analysis_v2.ipynb` | Data validation and EDA: funnel, representativeness, text quality, rating reliability, availability of success signals. | `outputs/{tables,figures}/eda/` |
| 03 | `03_sbert_network_main_v2.ipynb` | Masks character names, embeds the overviews (all-MiniLM-L6-v2), builds the mutual 20-NN network, detects communities (consensus Louvain) and computes centralities (exact betweenness). | `processed/network/`, `processed/features/` |
| 04 | `04_network_interpretation_v2.ipynb` | Interprets the network: distinctive words, labels for the 35 major communities, central, bridge and boundary films, communities vs genres. | `outputs/{tables,figures}/interpretation/`, labelled community file |
| 05 | `05_success_and_star_power_v2.ipynb` | Nested regressions (controls → plot network → star power → community) for revenue, rating and popularity; novelty sweet spot; community effects; robustness. | `outputs/{tables,figures}/success/`, `processed/features/modeling_table_success_star_power.csv` |
| 06 | `06_narrative_dynamics.ipynb` | Timing: crowding, foresight and competition; rising and fading narratives; Moran's *I*; out-of-sample prediction from plot neighbours. | `outputs/{tables,figures}/narrative_dynamics/` |
| 07 | `07_robustness_checks.ipynb` | Robustness: Benjamini–Hochberg correction of the plot, timing and community tests; foresight vs imitation (shorter foresight windows, does success attract similar stories?); the network rebuilt with k = 10/20/30 and from TF-IDF word vectors, with the Notebook 05 models and Moran's *I* re-run on each. | `outputs/{tables,figures}/robustness/` |
| 08 | `08_thesis_bricks.ipynb` | The thesis figures: eight presentation bricks built from the outputs above. It re-estimates nothing. | `outputs/thesis_bricks/` |

`tools/tidy_project.ipynb` moves files the pipeline no longer uses into an archive folder (see *Archive* below).

---

## Folder structure

```
raw/                      raw TMDB extract (Git LFS)
processed/
  core/                   corpora from Notebook 01 + corpus_manifest.json
  network/                nodes, edges, communities, node positions + network_manifest.json (Notebook 03–04)
  features/               sentence embeddings, modelling table, time-aware plot features (Notebooks 03, 05, 06)
outputs/
  tables/                 tables of every notebook; sub-folders per notebook:
    eda/ · interpretation/ · success/ · narrative_dynamics/ · corpus_audit/ · network_diagnostics/ · robustness/
                          (files directly in tables/ are written by Notebooks 01–05 under the names
                           later notebooks read, e.g. community_manual_labeling_template_labeled.csv)
  figures/                analysis figures, same sub-folders as tables/
  thesis_bricks/          Notebook 08: one folder per brick with figures/ (PNG + SVG), tables/ and text/
tools/                    housekeeping notebooks
_archive_2026-10/         earlier versions (v1 notebooks, old tables and figures), kept for reference
```

Every figure is saved together with a CSV of the numbers it shows. Each notebook also writes a short notes file with the key numbers, worded for the thesis text (e.g. `outputs/tables/success_star_power_notes.txt`, `outputs/thesis_bricks/08_structure_success/text/brick_08_notes.txt`).

---

## Thesis figures (Notebook 08)

| Brick | Reader question | Main-text figures |
|---|---|---|
| 1 · Data construction | How did 1.2 M titles become the network? | `fig_01_data_funnel`, `fig_02_year_coverage` |
| 2 · NLP similarity | How does a plot become a similarity? | `fig_01_neighbour_cards`, `fig_02_mutual_handshake`, `fig_03_similarity_scale` |
| 3 · Network construction | How do similarities become a network? | `fig_01_construction_flow`, `fig_05_edge_homophily` |
| 4 · Narrative communities | Which kinds of story exist? | `fig_01_major_communities`, `fig_02_community_index_table`, `fig_04_community_map` |
| 5 · Central positions | What does centrality mean here? | `fig_01_centrality_is_typicality`, `fig_03_central_films_table`, `fig_04_ego_network` |
| 6 · Bridging positions | Which films sit between narratives? | `fig_01_bridge_typology`, `fig_03_top_bridges`, `fig_04_bridge_pathway` |
| 7 · Success dimensions | Is success one thing? | `fig_01_outcome_correlations`, `fig_02_success_map` |
| 8 · Structure and success | How is plot position associated with success? | `fig_01_r2_by_block`, `fig_02_signal_grid`, `fig_03_similar_stories_similar_fates`, `fig_04_which_narratives_pay` |

The remaining figures of each brick are listed as appendix material in its `text/brick_NN_notes.txt`.

**Robustness (Notebook 07).** `outputs/tables/robustness/robustness_notes.txt` says which results survive a multiple-testing correction, whether the foresight result could be imitation, and how much the plot-position results depend on k and the text representation. The Brick 8 storyline uses these numbers when the folder exists. A second sentence-BERT model (all-mpnet-base-v2) runs automatically in Colab when a GPU runtime is selected.

---

## Running the project

1. Put the project folder on Google Drive as `MyDrive/thesis_movie_project` (or change `CFG["project_dir"]` in the first cell of each notebook).
2. Open a notebook in Colab and run all cells. Drive is mounted automatically.
3. Notebook 03 downloads the sentence-BERT model and the spaCy model `en_core_web_sm` on first use. A GPU speeds up the embedding step but is not required.

**Packages** (all preinstalled in Colab except where noted): pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, networkx, python-igraph, sentence-transformers, spacy. Notebooks 07 and 08 were tested with pandas 2.3, numpy 2.5, matplotlib 3.11, scipy 1.18, scikit-learn 1.9, statsmodels 0.15, networkx 3.7.

**Reproducibility.** Random steps use fixed seeds (`random_state = 42`). Each run writes manifests (`corpus_manifest.json`, `network_manifest.json`) with row counts and fingerprints, and later notebooks check them, so a stale input stops the run instead of silently changing results. One figure differs between machines: the t-SNE map in Brick 2 (`fig_05_embedding_map`) is laid out slightly differently by each scikit-learn version. Which communities sit together is stable; their position on the page is not.

---

## Archive

`_archive_2026-10/` holds everything from earlier stages: v1 notebooks (`notebooks/`, `embeddings/`), v1/v3 data tables, v1 figures and tables, and the v1 brick outputs. Each file keeps its original path inside the archive, and `_archive_2026-10/ARCHIVE_INDEX.csv` lists every move. Nothing in the pipeline reads from the archive.
