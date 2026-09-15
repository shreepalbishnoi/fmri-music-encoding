"""Step 4: render the per-voxel correlation and timbre/harmony/rhythm tuning
maps produced by 03_ridge_regression_tuning_maps.py as an interactive 3D view
and glass-brain figures.

Requires an fMRIPrep T1w reference plus map_correlation.nii.gz,
map_timbre_CLEAN.nii.gz, map_harmony_CLEAN.nii.gz, map_rhythm_CLEAN.nii.gz.
"""
from nilearn import plotting, image
from pathlib import Path

# --- CONFIG & LOADING ---
print("Loading cleaned results for visualization...")
SPACE = "space-MNI152NLin2009cAsym_res-2"
BASE_DIR = Path("../output/sub-001")
t1_path = BASE_DIR / "anat" / f"sub-001_{SPACE}_desc-preproc_T1w.nii.gz"

corr_img = image.load_img("map_correlation.nii.gz")
harmony_img = image.load_img("map_harmony_CLEAN.nii.gz")
timbre_img = image.load_img("map_timbre_CLEAN.nii.gz")
rhythm_img = image.load_img("map_rhythm_CLEAN.nii.gz")

# --- INTERACTIVE 3D VIEW ---
print("Generating 3D brain map (correlation)...")
view = plotting.view_img(
    corr_img,
    bg_img=image.load_img(str(t1_path)),
    threshold=0.15,
    title="Subject-001: Auditory Encoding Accuracy (r)",
    cmap='hot',
    black_bg=True
)
view.save_as_html("subject001_correlation_map.html")
print("Correlation map saved to HTML.")

# --- GLASS BRAIN (for slides/report) ---
print("Rendering glass brain for harmony...")
plotting.plot_glass_brain(
    harmony_img,
    display_mode='lyrz',
    colorbar=True,
    threshold=0.001,  # low threshold works because it's already masked
    title='Harmony Sensitivity (Segmented ROI)',
    plot_abs=False,
    cmap='OrRd'
)

# Repeat plotting.plot_glass_brain(...) for timbre_img / rhythm_img to compare
# specialization: the center of map_timbre_CLEAN vs. the outer edges of
# map_harmony_CLEAN.
#
# MRIcroGL RGB overlay setup:
#   map_harmony_CLEAN (Red)   | Min: 0.003, Max: 0.012
#   map_rhythm_CLEAN  (Green) | Min: 0.001, Max: 0.008
#   map_timbre_CLEAN  (Blue)  | Min: 0.003, Max: 0.012

plotting.show()
