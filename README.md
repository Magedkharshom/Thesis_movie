# The Semantic Topology of Cinema

Code and data for my Master's thesis in Data Science for Economics at the Università degli Studi di Milano
(supervisor: Prof. Matteo Zignani, co-supervisor: Prof. Sabrina Tiziana Gaito).

The project builds a network of about 19,000 films (1990–2024) from their plot summaries. Plots are embedded
with Sentence-BERT, films are linked to their mutual nearest neighbours, and communities of similar stories are
detected with consensus Louvain. The network positions and communities are then related to box-office revenue,
return on investment, popularity and ratings. All results are associations, not causal effects.

## Notebooks

Run them in order; each one reads the files written by the previous ones.

| Notebook | Content |
|---|---|
| 01 | Builds the film corpora from the raw TMDB extract |
| 02 | Data checks and exploratory analysis |
| 03 | Name masking, embeddings, network and communities |
| 04 | Interpretation of the communities |
| 05 | Regression models of success |
| 06 | Time-based measures, similar outcomes of similar films, prediction |
| 07 | Robustness checks |
| 08 | Figures used in the thesis |

The `revisions/` folder contains a few additional sensitivity checks that can be run on their own,
e.g. `python revisions/r04_selection.py`. Their results are saved in `outputs/tables/revisions/`.

## Data

The raw file (`TMDB_all_movies.csv`) is not included because of its size; see `raw/README.md` for the source.
The processed data in `processed/` are included, so everything from Notebook 02 onwards can be run without it.
Large files are stored with Git LFS.

## Running

The notebooks were written for Google Colab. Put the project folder on Google Drive as
`MyDrive/thesis_movie_project` (or change `CFG["project_dir"]` in the first cell) and run the cells in order.
Package versions are listed in `requirements.txt`. Random steps use fixed seeds.

Maged Saeed Abdo Mostafa Kharshom
