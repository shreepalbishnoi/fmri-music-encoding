"""Step 3b: normalize the timbre/harmony/rhythm tuning weights per category
and assign each significant voxel to whichever category dominates it
(winner-take-all), producing a single functional parcellation map.

Requires bold_all.npy + run_info.npy (step 1) and clip_features.npy +
events_all.csv (step 2). Re-runs the same ridge regression as step 3.

Output: map_correlation.nii.gz, map_WINNERS.nii.gz
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from nilearn import image
import gc
from pathlib import Path

# --- CONFIG & LOADING ---
print("Initializing normalized encoding model...")
SPACE = "space-MNI152NLin2009cAsym_res-2"
BASE_DIR = Path("../output/sub-001")
mask_path = BASE_DIR / "anat" / f"sub-001_{SPACE}_desc-brain_mask.nii.gz"

bold_all = np.load("bold_all.npy", mmap_mode='r')
run_info = np.load("run_info.npy", allow_pickle=True)
clip_features = np.load("clip_features.npy", allow_pickle=True).item()
events_all = pd.read_csv("events_all.csv")
mask_img = image.load_img(str(mask_path))

# --- CONSTRUCT DESIGN MATRIX (X) ---
n_trs, n_voxels = bold_all.shape
sample_key = next(iter(clip_features))
n_features = clip_features[sample_key].shape[1]

X = np.zeros((n_trs, n_features))
for _, row in events_all.iterrows():
    key = (row['genre'], int(row['track']), round(row['start'], 1), round(row['end'], 1))
    if key in clip_features:
        start_tr = int(row['global_onset_tr'])
        X[start_tr : start_tr + 10, :] = clip_features[key]

train_idx = [i for r in run_info if r['task'] == "Training" for i in range(r['start_tr'], r['end_tr'])]
test_idx = [i for r in run_info if r['task'] == "Test" for i in range(r['start_tr'], r['end_tr'])]
X_train, X_test = X[train_idx], X[test_idx]

# --- CHUNKED RIDGE & TUNING ---
chunk_size = 5000
all_corrs = []
t_raw, h_raw, r_raw = [], [], []

print(f"Processing {n_voxels} voxels...")

for i in range(0, n_voxels, chunk_size):
    end = min(i + chunk_size, n_voxels)
    Y_train = np.array(bold_all[train_idx, i:end])
    Y_test = np.array(bold_all[test_idx, i:end])

    model = Ridge(alpha=1000)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    for v in range(Y_test.shape[1]):
        r = pearsonr(Y_test[:, v], Y_pred[:, v])[0] if np.std(Y_test[:, v]) > 1e-6 else 0
        all_corrs.append(r)

    weights = model.coef_
    timbre_idx = [f + (d * 27) for d in range(6) for f in range(0, 13)]
    harmony_idx = [f + (d * 27) for d in range(6) for f in range(13, 25)]
    rhythm_idx = [f + (d * 27) for d in range(6) for f in [25, 26]]

    t_raw.extend(np.mean(np.abs(weights[:, timbre_idx]), axis=1))
    h_raw.extend(np.mean(np.abs(weights[:, harmony_idx]), axis=1))
    r_raw.extend(np.mean(np.abs(weights[:, rhythm_idx]), axis=1))

    del Y_train, Y_test, Y_pred
    gc.collect()

# --- NORMALIZED WINNER-TAKE-ALL ---
print("\nNormalizing features and calculating winners...")
t_arr, h_arr, r_arr = np.array(t_raw), np.array(h_raw), np.array(r_raw)

def normalize(arr):
    if (arr.max() - arr.min()) == 0:
        return arr
    return (arr - arr.min()) / (arr.max() - arr.min())

t_norm, h_norm, r_norm = normalize(t_arr), normalize(h_arr), normalize(r_arr)

# 1 = Timbre, 2 = Harmony, 3 = Rhythm
stacked = np.vstack([t_norm, h_norm, r_norm]).T
winner_indices = np.argmax(stacked, axis=1) + 1

# Restrict to voxels with reliable prediction accuracy (r > 0.15)
r_mask = (np.array(all_corrs) > 0.15).astype(float)
winner_map = winner_indices * r_mask

def save_as_nifti(data_array, filename):
    vol = np.zeros(mask_img.shape)
    vol[mask_img.get_fdata().astype(bool)] = np.array(data_array)
    img = image.new_img_like(mask_img, vol)
    img.to_filename(filename)

save_as_nifti(all_corrs, "map_correlation.nii.gz")
save_as_nifti(winner_map, "map_WINNERS.nii.gz")

print("Done. Created 'map_WINNERS.nii.gz'.")
print(f"Stats: Timbre wins: {np.sum(winner_map==1)}, Harmony wins: {np.sum(winner_map==2)}, Rhythm wins: {np.sum(winner_map==3)}")
