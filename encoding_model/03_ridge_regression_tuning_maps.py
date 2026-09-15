"""Step 3: build the design matrix from the delayed music features, run
chunked ridge regression per voxel, and save per-voxel tuning maps for
timbre / harmony / rhythm (masked to voxels with prediction accuracy r > 0.15).

Requires bold_all.npy + run_info.npy (step 1) and clip_features.npy +
events_all.csv (step 2).

Outputs: map_correlation.nii.gz, map_timbre_CLEAN.nii.gz,
map_harmony_CLEAN.nii.gz, map_rhythm_CLEAN.nii.gz
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from nilearn import image
import gc
from pathlib import Path

# --- CONFIG & LOADING ---
print("Initializing masked encoding model...")
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

    # Correlation
    for v in range(Y_test.shape[1]):
        r = pearsonr(Y_test[:, v], Y_pred[:, v])[0] if np.std(Y_test[:, v]) > 1e-6 else 0
        all_corrs.append(r)

    # Tuning: mean |beta weight| per feature category (27 features x 6 delays)
    weights = model.coef_
    timbre_idx = [f + (d * 27) for d in range(6) for f in range(0, 13)]
    harmony_idx = [f + (d * 27) for d in range(6) for f in range(13, 25)]
    rhythm_idx = [f + (d * 27) for d in range(6) for f in [25, 26]]

    t_raw.extend(np.mean(np.abs(weights[:, timbre_idx]), axis=1))
    h_raw.extend(np.mean(np.abs(weights[:, harmony_idx]), axis=1))
    r_raw.extend(np.mean(np.abs(weights[:, rhythm_idx]), axis=1))

    del Y_train, Y_test, Y_pred
    gc.collect()

# --- APPLY FUNCTIONAL MASKING (r > 0.15) ---
print("\nApplying auditory significance mask (r > 0.15) to tuning maps...")
r_mask = (np.array(all_corrs) > 0.15).astype(float)
t_clean = np.array(t_raw) * r_mask
h_clean = np.array(h_raw) * r_mask
r_clean = np.array(r_raw) * r_mask

# --- SAVE AS NIFTI ---
def save_as_nifti(data_array, filename):
    vol = np.zeros(mask_img.shape)
    vol[mask_img.get_fdata().astype(bool)] = np.array(data_array)
    img = image.new_img_like(mask_img, vol)
    img.to_filename(filename)

save_as_nifti(all_corrs, "map_correlation.nii.gz")
save_as_nifti(t_clean, "map_timbre_CLEAN.nii.gz")
save_as_nifti(h_clean, "map_harmony_CLEAN.nii.gz")
save_as_nifti(r_clean, "map_rhythm_CLEAN.nii.gz")

print("Done. Use the '_CLEAN' files in MRIcroGL or 04_visualize_tuning_maps.py.")
