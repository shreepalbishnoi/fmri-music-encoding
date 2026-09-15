import os
import numpy as np
import nibabel as nib
import gc
from nilearn.masking import compute_epi_mask, apply_mask
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Directory containing the raw ds003720 (OpenNeuro Music Genre) download
DATASET_DIR = os.environ.get("DS003720_DIR", os.path.join(ROOT, "..", "datasets", "ds003720-download"))
FMRI_DIR = os.path.join(DATASET_DIR, "sub-001", "func")
FEAT_DIR = os.path.join(ROOT, "features")

# Runs for modeling
TRAIN_RUNS = [f"sub-001_task-Training_run-{i:02d}" for i in range(1, 13)]
TEST_RUNS = [f"sub-001_task-Test_run-{i:02d}" for i in range(1, 7)]

def apply_hrf_delay(F, delays=[1, 2, 3, 4, 5]):
    # F: [T, N]
    T, N = F.shape
    F_lagged = []
    for d in delays:
        F_d = np.zeros_like(F)
        if d < T:
            F_d[d:, :] = F[:-d, :]
        F_lagged.append(F_d)
    return np.concatenate(F_lagged, axis=1)

def load_features(runs, pca_model=None):
    # Returns Cochlear, MTF (PCA reduced + HRF), MFCC
    coch_list, mtf_list, mfcc_list = [], [], []
    for r in runs:
        f_path = os.path.join(FEAT_DIR, f"{r}_audio_features.npz")
        data = np.load(f_path)
        coch_list.append(data['cochlear'])
        mtf_list.append(data['mtf'])
        mfcc_list.append(data['mfcc'])
        
    Coch = np.concatenate(coch_list, axis=0)
    Mtf = np.concatenate(mtf_list, axis=0)
    Mfcc = np.concatenate(mfcc_list, axis=0)
    
    # Standardize early
    Coch = StandardScaler().fit_transform(Coch)
    Mtf = StandardScaler().fit_transform(Mtf)
    Mfcc = StandardScaler().fit_transform(Mfcc)
    
    if pca_model is None: # Training
        pca_model = PCA(n_components=302)
        Mtf = pca_model.fit_transform(Mtf)
    else:
        Mtf = pca_model.transform(Mtf)
        
    Coch_final = apply_hrf_delay(Coch)
    Mtf_final = apply_hrf_delay(Mtf)
    Mfcc_final = apply_hrf_delay(Mfcc)
    
    return Coch_final, Mtf_final, Mfcc_final, pca_model

def get_mask(runs):
    print("Computing brain mask...")
    sample_path = os.path.join(FMRI_DIR, f"{runs[0]}_bold.nii")
    mask_img = compute_epi_mask(nib.load(sample_path))
    mask_data = mask_img.get_fdata(dtype=np.float32) > 0
    print("Brain mask computed. Non-zero voxels:", mask_data.sum())
    return mask_data

def train_ridge(F_tr, mask_data, alpha=1000.0):
    N = F_tr.shape[1]
    V = mask_data.sum()
    XTX = np.zeros((N, N), dtype=np.float32)
    XTY = np.zeros((N, V), dtype=np.float32)
    
    start = 0
    for r in TRAIN_RUNS:
        print(f"Loading {r} for training...")
        img_path = os.path.join(FMRI_DIR, f"{r}_bold.nii")
        data = nib.load(img_path).get_fdata(dtype=np.float32)
        Y_run = data[mask_data].T
        del data
        
        # Audio length is 600s = 400 TRs. fMRI might be longer (e.g. 410). Crop to 400.
        T_audio = 400
        Y_run = Y_run[:T_audio]
        
        # Z-score natively
        Y_run -= Y_run.mean(axis=0)
        std = Y_run.std(axis=0)
        std[std == 0] = 1.0
        Y_run /= std
        
        T_run = Y_run.shape[0]
        X_run = F_tr[start : start + T_run]
        
        XTX += X_run.T @ X_run
        XTY += X_run.T @ Y_run
        start += T_run
        
    XTX += alpha * np.eye(N)
    print("Solving linear system (in batches)...")
    W = np.zeros_like(XTY)
    batch_size = 20000
    for v_start in range(0, V, batch_size):
        W[:, v_start:v_start+batch_size] = np.linalg.solve(XTX, XTY[:, v_start:v_start+batch_size])
    return W

def eval_ridge(F_ts, W, mask_data):
    V = mask_data.sum()
    sum_y = np.zeros(V, dtype=np.float32)
    sum_y2 = np.zeros(V, dtype=np.float32)
    sum_ypred = np.zeros(V, dtype=np.float32)
    sum_ypred2 = np.zeros(V, dtype=np.float32)
    sum_y_ypred = np.zeros(V, dtype=np.float32)
    
    start = 0
    N_samples = 0
    for r in TEST_RUNS:
        print(f"Loading {r} for testing...")
        img_path = os.path.join(FMRI_DIR, f"{r}_bold.nii")
        data = nib.load(img_path).get_fdata(dtype=np.float32)
        Y_run = data[mask_data].T
        del data
        
        T_audio = 400
        Y_run = Y_run[:T_audio]
        
        # Z-score natively
        Y_run -= Y_run.mean(axis=0)
        std = Y_run.std(axis=0)
        std[std == 0] = 1.0
        Y_run /= std
        
        T_run = Y_run.shape[0]
        X_run = F_ts[start : start + T_run]
        Y_pred = X_run @ W
        
        sum_y += Y_run.sum(axis=0)
        sum_y2 += (Y_run**2).sum(axis=0)
        sum_ypred += Y_pred.sum(axis=0)
        sum_ypred2 += (Y_pred**2).sum(axis=0)
        sum_y_ypred += (Y_run * Y_pred).sum(axis=0)
        
        start += T_run
        N_samples += T_run
        
    # Compute corr
    mean_y = sum_y / N_samples
    mean_ypred = sum_ypred / N_samples
    cov = sum_y_ypred - N_samples * mean_y * mean_ypred
    var_y = sum_y2 - N_samples * (mean_y**2)
    var_ypred = sum_ypred2 - N_samples * (mean_ypred**2)
    
    denom = np.sqrt(var_y * var_ypred)
    corrs = np.zeros(V, dtype=np.float32)
    valid = denom > 0
    corrs[valid] = cov[valid] / denom[valid]
    
    return corrs

def main():
    print("Loading Training Features...")
    F_coch_tr, F_mtf_tr, F_mfcc_tr, pca_model = load_features(TRAIN_RUNS)
    print("Loading Validation Features...")
    F_coch_ts, F_mtf_ts, F_mfcc_ts, _ = load_features(TEST_RUNS, pca_model)
    
    mask_data = get_mask(TRAIN_RUNS)
    
    print("Training/Evaluating Cochlear...")
    W_coch = train_ridge(F_coch_tr, mask_data)
    corrs_coch = eval_ridge(F_coch_ts, W_coch, mask_data)
    print(f"==> Cochlear Mean Pearson Correlation: {np.mean(corrs_coch):.4f}")
    del W_coch; gc.collect()
    
    print("Training/Evaluating MTF...")
    W_mtf = train_ridge(F_mtf_tr, mask_data)
    corrs_mtf = eval_ridge(F_mtf_ts, W_mtf, mask_data)
    print(f"==> MTF Mean Pearson Correlation: {np.mean(corrs_mtf):.4f}")
    del W_mtf; gc.collect()
    
    print("Training/Evaluating MFCC...")
    W_mfcc = train_ridge(F_mfcc_tr, mask_data)
    corrs_mfcc = eval_ridge(F_mfcc_ts, W_mfcc, mask_data)
    print(f"==> MFCC Mean Pearson Correlation: {np.mean(corrs_mfcc):.4f}")
    del W_mfcc; gc.collect()
    
    out_path = os.path.join(ROOT, "results", "evaluation_results.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path,
                        cochlear=corrs_coch,
                        mtf=corrs_mtf,
                        mfcc=corrs_mfcc)

    print(f"Done! Results saved to {out_path}")


if __name__ == "__main__":
    main()
