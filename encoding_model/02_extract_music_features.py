"""Step 2: align each run's stimulus events to global TR indices, then extract
timbre (MFCC), harmony (chroma), and rhythm (RMS + spectral centroid) features
per music clip from the GTZAN genre dataset, with 6 hemodynamic-lag delays.

Requires run_info.npy from 01_extract_bold_confound_regression.py, a local
copy of the GTZAN "genres_original" audio, and the ds003720 raw events.tsv
files (see docs/review-2-encoding-model-and-results.pdf, "Week 5" section).

Outputs: events_all.csv, clip_features.npy
"""
import pandas as pd
import numpy as np
import librosa
from pathlib import Path

# --- CONFIG ---
GTZAN_DIR = Path("../genres_original")
RAW_DIR = Path("../ds003720/sub-001/func")
run_info = np.load("run_info.npy", allow_pickle=True)

TR = 1.5
SR = 22050
N_TRS_PER_CLIP = 10
N_DELAYS = 5  # 6 total taps (0..5 * TR) covering ~7.5s of hemodynamic lag

# --- STACK EVENTS ---
print("Aligning event timings...")
all_events = []
for r in run_info:
    ev_path = RAW_DIR / f"sub-001_task-{r['task']}_run-{r['run']}_events.tsv"
    if not ev_path.exists():
        print(f"Missing events for {r['task']} run {r['run']}")
        continue

    ev = pd.read_csv(ev_path, sep='\t')
    ev["genre"] = ev["genre"].str.replace("'", "").str.strip()
    ev["task"], ev["run"] = r["task"], r["run"]
    ev["global_onset_tr"] = (ev["onset"] / TR).astype(int) + r["start_tr"]
    all_events.append(ev)

if not all_events:
    raise ValueError("No event files found! Check RAW_DIR path.")

events_all = pd.concat(all_events, ignore_index=True)
events_all.to_csv("events_all.csv", index=False)
print(f"events_all.csv saved with {len(events_all)} entries.")

# --- MULTI-FEATURE EXTRACTION ---
print("\nExtracting hierarchical music features...")
clip_features = {}
unique_clips = events_all[["genre", "track", "start", "end"]].drop_duplicates()

for _, row in unique_clips.iterrows():
    genre, track = row["genre"], int(row["track"])
    s_start, s_end = round(row["start"], 1), round(row["end"], 1)

    # GTZAN folder naming fix (e.g. 'hip-hop' folder is often 'hiphop')
    g_folder = genre.replace('-', '')
    a_path = GTZAN_DIR / g_folder / f"{g_folder}.{track-1:05d}.wav"
    if not a_path.exists():
        continue

    try:
        audio, _ = librosa.load(str(a_path), sr=SR, offset=s_start, duration=TR * N_TRS_PER_CLIP)

        feats = []
        samples_per_tr = int(TR * SR)
        for i in range(N_TRS_PER_CLIP):
            seg = audio[i*samples_per_tr : (i+1)*samples_per_tr]
            if len(seg) < samples_per_tr:
                seg = np.pad(seg, (0, samples_per_tr - len(seg)))

            # 1. Timbre: MFCC (13 features)
            mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=13).mean(axis=1)
            # 2. Harmony: chroma (12 features, one per semitone)
            chroma = librosa.feature.chroma_stft(y=seg, sr=SR).mean(axis=1)
            # 3. Rhythm/dynamics: RMS energy (1 feature)
            rms = librosa.feature.rms(y=seg).mean(axis=1)
            # 4. Spectral brightness: centroid (1 feature)
            centroid = librosa.feature.spectral_centroid(y=seg, sr=SR).mean(axis=1)

            # 27 features per TR
            combined_tr = np.concatenate([mfcc, chroma, rms, centroid])
            feats.append(combined_tr)

        feats = np.array(feats)  # (10 TRs, 27)

        # Temporal delays to account for hemodynamic lag
        delayed = [feats]
        for d in range(1, N_DELAYS + 1):
            shft = np.zeros_like(feats)
            shft[d:] = feats[:-d]
            delayed.append(shft)

        # Final vector: 10 TRs x (27 features * 6 delays) = 162 total features per TR
        clip_features[(genre, track, s_start, s_end)] = np.hstack(delayed)

    except Exception as e:
        print(f"Error in {a_path.name}: {e}")
        continue

np.save("clip_features.npy", clip_features)
print(f"Extracted {len(clip_features)} feature sets (27 base features per TR).")
