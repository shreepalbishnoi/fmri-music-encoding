import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

SR = 22050
N_FILTERS = 128
FMIN = 100
FMAX = 8000
WIN_LEN = int(0.025 * SR)   # 25 ms
HOP_LEN = int(0.010 * SR)   # 10 ms
TR = 1.5

def compute_cochleogram(y):
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=WIN_LEN, hop_length=HOP_LEN,
        n_mels=N_FILTERS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    return np.log1p(S)

def downsample_to_TR(feat):
    # feat shape: [features, frames]
    frame_time = HOP_LEN / SR
    frames_per_TR = int(TR / frame_time)
    T = feat.shape[1] // frames_per_TR
    out = []
    for i in range(T):
        chunk = feat[:, i*frames_per_TR:(i+1)*frames_per_TR]
        out.append(chunk.mean(axis=1))
    return np.array(out)  # [T, features]

def compute_mtf(cochleo):
    freq_bins, time_bins = cochleo.shape
    omega = np.array([2.8, 4.0, 5.7, 8.0, 11.3, 16.0, 22.6, 32.0, 45.3, 64.0])
    Omega = np.array([0.35, 0.50, 0.71, 1.0, 1.41, 2.0, 2.83, 4.0, 5.66, 8.0])
    
    mtf_features = []
    for w in omega:
        fft_time = np.fft.fft(cochleo, axis=1)
        freqs = np.fft.fftfreq(time_bins, d=HOP_LEN/SR)
        mask = np.abs(freqs - w) < w*0.2
        filt = fft_time * mask
        temp_filtered = np.real(np.fft.ifft(filt, axis=1))
        
        for W in Omega:
            fft_freq = np.fft.fft(temp_filtered, axis=0)
            freqs_f = np.fft.fftfreq(freq_bins)
            mask_f = np.abs(freqs_f - W) < W*0.2
            filt_f = fft_freq * mask_f[:, None]
            spec_filtered = np.real(np.fft.ifft(filt_f, axis=0))
            
            energy = spec_filtered**2
            
            # Average within 20 nonoverlapping frequency ranges.
            energy_bands = np.array([np.mean(b, axis=0) for b in np.array_split(energy, 20, axis=0)])
            mtf_features.append(energy_bands) # shape: [20, time]
            
    mtf_features = np.concatenate(mtf_features, axis=0) # [10x10x20 = 2000, time]
    return np.log1p(mtf_features)

def compute_mfcc(y):
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=WIN_LEN, hop_length=HOP_LEN,
        n_mels=N_FILTERS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=12)
    return mfcc

def extract_features(audio_path):
    print(f"Loading {audio_path}")
    y, sr = librosa.load(audio_path, sr=SR)
    
    print("Computing Cochlear...")
    cochleo = compute_cochleogram(y)
    cochleo_TR = downsample_to_TR(cochleo)
    
    print("Computing MTF...")
    mtf = compute_mtf(cochleo)
    mtf_TR = downsample_to_TR(mtf)
    
    print("Computing MFCC...")
    mfcc = compute_mfcc(y)
    mfcc_TR = downsample_to_TR(mfcc)
    
    return cochleo_TR, mtf_TR, mfcc_TR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(ROOT, "recons_audio")
OUT_DIR = os.path.join(ROOT, "features")
os.makedirs(OUT_DIR, exist_ok=True)

audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])

for f in audio_files:
    out_path = os.path.join(OUT_DIR, f.replace(".wav", "_features.npz"))
    if os.path.exists(out_path):
        print(f"Skipping {f} (already exists)")
        continue
        
    cochleo_TR, mtf_TR, mfcc_TR = extract_features(os.path.join(AUDIO_DIR, f))
    np.savez_compressed(out_path, cochlear=cochleo_TR, mtf=mtf_TR, mfcc=mfcc_TR)
    print(f"Saved features for {f} -> mtf shape: {mtf_TR.shape}")
