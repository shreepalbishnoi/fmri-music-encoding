import os
import nibabel as nib

# Directory containing the raw ds003720 (OpenNeuro Music Genre) download,
# e.g. .../ds003720-download/sub-001/func
DATASET_DIR = os.environ.get("DS003720_DIR", "../datasets/ds003720-download")

img_path = os.path.join(DATASET_DIR, "sub-001", "func", "sub-001_task-Training_run-01_bold.nii")
img = nib.load(img_path)
data = img.get_fdata()
print("Data shape:", data.shape)
print("Data type:", data.dtype)
