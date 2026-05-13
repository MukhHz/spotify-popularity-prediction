# Predicting Spotify Track Popularity from Audio Features

> 🏆 **Best Presentation** — Stat 451 (Intro to Machine Learning and Statistical Pattern Classification), UW–Madison, Spring 2026
> Group 12

How well can audio features predict how popular a song will be on Spotify? We pulled the Spotify Tracks Dataset (~114K tracks across 125 genres), cleaned it down to ~88K real songs, and built four regression models — OLS, Ridge, Lasso, and a Decision Tree — to predict the `popularity` score from features like danceability, energy, loudness, valence, tempo, and genre.

## Results

| Model | R² (validation) | RMSE |
|---|---|---|
| OLS Linear | 0.336 | 16.74 |
| Ridge | 0.336 | 16.74 |
| Lasso | 0.336 | 16.74 |
| Decision Tree | 0.335 | 16.74 |
| **OLS on held-out test set** | **~0.35** | **~16.5** |

All four models landed in the same neighborhood — adding complexity didn't help, which itself is a finding. Audio features and genre together explain about a third of the variation in popularity. The rest is driven by things outside the dataset: marketing, artist reputation, release timing, playlist placement, cultural moment.

**What mattered most:**
- **Genre** was the single strongest predictor across every model. Including genre dummies meaningfully improved performance over audio features alone.
- **Top audio features:** danceability and loudness (positive), instrumentalness and speechiness (negative). Valence, acousticness, and energy also contributed.
- The decision tree surfaced the same audio importances as the linear models (loudness, duration, danceability, acousticness on top), with specific genre dummies — iranian, romance, k-pop — leading the overall feature importance ranking.

## Approach

**Cleaning** (`notebooks/data_cleaning.ipynb`)
- Started with 114K rows × 21 columns from Kaggle
- Dropped 1 fully-null row, deduplicated on `track_id` (same song often appears under multiple genres)
- Filtered out non-music: `speechiness > 0.66` (audiobooks, podcasts, poetry)
- Removed tracks with `duration < 30s` or `tempo == 0` (clips, errors)
- Encoded `explicit` as 0/1, converted `duration_ms` to `duration_min`, dropped identifier columns (`track_id`, `artists`, `album_name`, `track_name`)
- **Final cleaned dataset: 88,704 rows**

**Modeling** (`notebooks/projectstat451.ipynb`)
- 80/10/10 train/validation/test split (random_state=42)
- One-hot encoded `track_genre` (124 dummies with drop_first)
- Standardized features for linear models; raw values for the decision tree
- Tuned hyperparameters with 5-fold `GridSearchCV` on the training set, scoring R²
- Evaluated on the validation set, then retrained the best model (OLS) on train+val combined and scored once on the held-out test set
- Compared coefficient signs across OLS, Ridge, and Lasso to confirm agreement on direction

## Repository structure

```
.
├── README.md
├── notebooks/
│   ├── data_cleaning.ipynb           # Cleaning pipeline → spotify_cleaned.csv
│   └── projectstat451.ipynb          # EDA, modeling, evaluation (includes cleaning at top)
└── report/
    └── Stat_451_Group_12_Project_Report.pdf
```

Note: the modeling notebook re-runs the cleaning steps at the top before doing EDA and modeling. You can run either notebook end-to-end on the raw `dataset.csv`.

## Data

Spotify Tracks Dataset from Kaggle: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Download `dataset.csv` and place it in the repo root before running the notebooks. The data isn't committed here.

## Reproducing

```bash
# 1. Download dataset.csv from Kaggle into the repo root
# 2. Run cleaning
jupyter notebook notebooks/data_cleaning.ipynb
# Produces spotify_cleaned.csv

# 3. Run EDA + modeling
jupyter notebook notebooks/projectstat451.ipynb
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## Limitations

- The dataset only captures audio features and genre — none of the external factors (marketing, artist popularity, release context) that probably drive the other ~65% of variation.
- Spotify-derived data has regional and genre biases baked in.
- We treated `popularity` as continuous, but it's a 0–100 integer with discrete jumps (the histogram in the report shows a sharp spike at 0).

Future work would pull in artist-level features (follower counts, prior chart performance) and compare across platforms (Apple Music, Billboard).

## Group 12

Arianna Barbosa · Eirfann Danish Bin Farul Muzri · Greta Fogel · Levi Hellenbrand · Mukhriz Hazli Bin Mohamad · Lexi Rush

Stat 451: Introduction to Machine Learning and Statistical Pattern Classification — UW–Madison, Spring 2026. Won Best Presentation.
