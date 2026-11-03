# Raw data

The raw TMDB extract is not stored in this repository because of its size (about 730 MB).

The analysis uses a single file, `TMDB_all_movies.csv` (1,192,309 records), from the Kaggle dataset
**TMDB Movies Daily Updates**: https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates

The dataset is updated daily. The version used in the thesis was downloaded on [DOWNLOAD DATE];
a newer download will contain more films and will not reproduce the corpus exactly.

To rerun Notebook 01, place `TMDB_all_movies.csv` in this folder. All other notebooks start from the
processed files in `processed/`, which are included.
