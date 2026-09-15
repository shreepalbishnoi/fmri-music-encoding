"""Step 4b: render the winner-take-all functional parcellation map
(map_WINNERS.nii.gz from 03b_winner_take_all_map.py) as a glass brain.
"""
from nilearn import plotting, image
from pathlib import Path

print("Loading winner-take-all map...")
SPACE = "space-MNI152NLin2009cAsym_res-2"
BASE_DIR = Path("../output/sub-001")
t1_path = BASE_DIR / "anat" / f"sub-001_{SPACE}_desc-preproc_T1w.nii.gz"

winner_img = image.load_img("map_WINNERS.nii.gz")

print("Rendering functional parcellation...")
plotting.plot_glass_brain(
    winner_img,
    display_mode='lyrz',
    colorbar=True,
    threshold=0.5,  # shows categories 1, 2, and 3
    title='Functional Specialization: Timbre (1), Harmony (2), Rhythm (3)',
    cmap='Paired',
    plot_abs=False
)

# MRIcroGL steps for the same view:
#   1. Load the T1w background.
#   2. Add map_WINNERS.nii.gz as an overlay.
#   3. Set Color Map to "Qualitative" or "NIH".
#   4. Set Darkest: 0.5, Brightest: 3.5.
#   5. Observe the boundaries between the 3 colors in the temporal lobe.

plotting.show()
