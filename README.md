# Mapping Acoustic Signal Features to Neural Activation Patterns During Music Listening Using fMRI

Biomedical Image Analysis course project. We model how the brain's BOLD
response during music listening relates to acoustic features of the music
being heard, using voxel-wise encoding models.

**Team:** Shreepal (EC23B1107), Shriya (EC23I1008), Ashok (EC23I1014)

## Dataset

[OpenNeuro ds003720](https://openneuro.org/datasets/ds003720/versions/1.0.0) —
participants listened to 30s clips from multiple music genres (classical,
jazz, rock, electronic, ...) while whole-brain BOLD fMRI was recorded.
Preprocessing was done with fMRIPrep (anatomical + functional pipelines,
confound estimation for motion/CSF/white-matter/global-signal regression).

The raw dataset and its fMRIPrep `output/` are not included in this repo
(large, and freely re-downloadable) — scripts expect them one level above
the repo, or via the `DS003720_DIR` environment variable. See
[`docs/review-2-encoding-model-and-results.pdf`](docs/review-2-encoding-model-and-results.pdf)
for the exact preprocessing steps.

## Two encoding pipelines

The project went through two iterations, both kept here since they represent
different feature-engineering approaches to the same question:

- **`encoding_model/`** — the pipeline behind the final presented results.
  Audio features are extracted per music clip (from GTZAN genre clips aligned
  to each run's `events.tsv`): 13 MFCCs (timbre), 12 chroma bins (harmony),
  RMS energy + spectral centroid (rhythm/dynamics) — 27 features x 6
  hemodynamic-lag delays = 162 dims. A ridge regression is fit per voxel
  (~238,000 voxels) to predict BOLD from these features, evaluated with
  per-voxel Pearson correlation. Best voxels reached **r = 0.33** in
  Heschl's Gyrus / Superior Temporal Gyrus. Run in order:
  1. `01_extract_bold_confound_regression.py` — load fMRIPrep BOLD, regress
     out motion/CSF/WM/global-signal confounds, concatenate all runs.
  2. `02_extract_music_features.py` — align stimulus events to global TRs,
     extract timbre/harmony/rhythm features with temporal delays.
  3. `03_ridge_regression_tuning_maps.py` — voxel-wise ridge regression,
     saves correlation + per-category (timbre/harmony/rhythm) tuning maps.
  3b. `03b_winner_take_all_map.py` — same regression, but assigns each
     significant voxel to whichever category (timbre/harmony/rhythm) drives
     it most (winner-take-all functional parcellation).
  4. `04_visualize_tuning_maps.py` / `04b_visualize_winner_map.py` — glass
     brain and interactive 3D renders of the maps above.

- **`scripts/`** — an earlier/parallel exploration using full-run stimulus
  audio (`recons_audio/`, see below) instead of per-clip GTZAN audio, with a
  richer auditory model: log-mel cochleogram (128 channels), a Modulation
  Transfer Function representation (2000-d, PCA-reduced to 302), and MFCC
  (12 channels), each downsampled to the fMRI TR (1.5s) and lagged by 5 HRF
  delays. Ridge regression is solved directly via the normal equations
  (`XᵀX`/`XᵀY`) per feature model, batched over voxels.
  - `check_nii.py` — sanity-check a raw BOLD NIfTI file's shape/dtype.
  - `extract_features.py` — compute cochlear/MTF/MFCC features from
    `recons_audio/*.wav`, saved to `features/*.npz`.
  - `train_model.py` — ridge regression + evaluation per feature model,
    saves `results/evaluation_results.npz`.
  - `plot_results.py` — bar chart comparing the three feature models on
    their top 1000 voxels, saved to `results/model_performance.png`.
  - `build_final_notebook.py` — (re)generates
    `notebooks/final_encoding_pipeline.ipynb` from source.

`notebooks/mri_test.ipynb` and `notebooks/final_encoding_pipeline.ipynb` walk
through the `scripts/` pipeline end-to-end with inline visualizations.

## `recons_audio/`

`extract_features.py` and the notebooks read full-length (~600s) per-run
stimulus audio from `recons_audio/*.wav`. These were reconstructed by
concatenating the individual 30s GTZAN clips presented in each run, in the
order and at the onsets given by that run's `events.tsv` (the same event
files `encoding_model/02_extract_music_features.py` reads directly per-clip).
The concatenation script itself wasn't among the recovered project files —
regenerate `recons_audio/*.wav` by stitching each run's clips per its
`events.tsv`, or ask the team for the cached files. This folder is excluded
from git (~455MB of `.wav` data); see `.gitignore`.

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Some `encoding_model/` and `scripts/` files also expect `nilearn`'s pulled
MNI templates (downloaded automatically on first use) and, for
`encoding_model/`, an fMRIPrep `output/sub-001` directory and a local GTZAN
`genres_original/` copy alongside the repo (see path comments at the top of
each script).

## Docs

- [`docs/review-1-dataset-and-preprocessing.pdf`](docs/review-1-dataset-and-preprocessing.pdf) — early-stage presentation: dataset selection, journal review, planned preprocessing pipeline.
- [`docs/review-2-encoding-model-and-results.pdf`](docs/review-2-encoding-model-and-results.pdf) — final presentation: fMRIPrep pipeline, feature engineering, ridge encoding results, timbre/harmony/rhythm tuning maps, genre-wise analysis.
- [`docs/reference-paper.pdf`](docs/reference-paper.pdf) — reference journal publication the project's methodology is based on.

## Results

`results/evaluation_results.npz` and `results/model_performance.png` hold the
`scripts/` pipeline's per-voxel correlations for the cochlear/MTF/MFCC
feature models (top-1000-voxel comparison). The `encoding_model/` pipeline's
NIfTI tuning maps (`map_correlation.nii.gz`, `map_timbre_CLEAN.nii.gz`, etc.)
are generated locally when the scripts are run against the full fMRIPrep
output and are not checked in.

## Not included

`Bio_medical/` (course-assignment sample images for an unrelated Medical
Image Processing exercise — T3/T5 tutorials) is kept locally but excluded
from this repo; it isn't part of this project.
