import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = np.load(os.path.join(ROOT, "results", "evaluation_results.npz"))
coch_r = data['cochlear']
mtf_r = data['mtf']
mfcc_r = data['mfcc']

# Drop nans
coch_r = coch_r[~np.isnan(coch_r)]
mtf_r = mtf_r[~np.isnan(mtf_r)]
mfcc_r = mfcc_r[~np.isnan(mfcc_r)]

# To see predictive power, let's look at the top 1000 voxels 
# (which generally fall in primary sensory context like auditory cortex)
top_n = 1000

coch_top = np.sort(coch_r)[-top_n:]
mtf_top = np.sort(mtf_r)[-top_n:]
mfcc_top = np.sort(mfcc_r)[-top_n:]

fig, ax = plt.subplots(figsize=(8, 5))
models = ['Cochlear', 'MTF (302 PCA)', 'MFCC (12 ch)']
means = [np.mean(coch_top), np.mean(mtf_top), np.mean(mfcc_top)]
stds = [np.std(coch_top), np.std(mtf_top), np.std(mfcc_top)]

bars = ax.bar(models, means, yerr=stds, capsize=5, color=['#4dc9f6', '#f67019', '#f53794'], alpha=0.8)
ax.set_ylabel('Pearson Correlation (R)')
ax.set_title('Top 1000 Voxel Predictive Accuracy by Feature Model')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.3f}", ha='center', va='bottom')

plt.tight_layout()
out_path = os.path.join(ROOT, "results", "model_performance.png")
plt.savefig(out_path, dpi=200)
print(f"Plot saved to {out_path}")
