import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from Initiate_PSiteDataset import PSiteDataset
from AccessCorrectFolders import access_folders, abbreviate_embed_names

'''
Create a heatmap showing the mean average over all P-sites for a specified field size surrounding the putative P-site 
for all embedded channels (or part of in case the number is too large to visualize).

Execute by calling `python Heatmap_Psites.py <embed_type> <P-site> <radius>
- embed_type = onehot, onehot+DSSP, esm, carp or alphafold
- P-site = ST or Y
- radius: radius of the receptive field surrounding the P-site (e.g. 10 --> 10+S/T/Y+10 = 21)
'''

embed_type, psite, radius = sys.argv[1], sys.argv[2], int(sys.argv[3])
embed_abbrev = abbreviate_embed_names(embed_type)

rep_dirs, lab_files = access_folders(embed_type, psite)


# Create the datasets
datasets = []
for i in range(3):
    datasets.append(PSiteDataset(data_rep_dir=rep_dirs[i], labels_fasta=lab_files[i], field_radius=radius))

rf_pos, rf_neg = [], []

title_part = 'Ser/Thr' if psite == 'ST' else 'Tyr'

# Extract the representations of the receptive fields and separate based on whether the central site is a P-site or not
for ds in datasets:
    for X, y, seq, *dssp in ds:
        if y == 1:
            rf_pos.append(X)
        else:
            rf_neg.append(X)

# Convert the lists of representations into numpy arrays and substract the non-P-sites from P-sites
rf_pos_tensor = torch.stack(tuple(rf_pos))
rf_pos_combined = torch.mean(rf_pos_tensor, dim=0)
rf_pos_array = np.array(rf_pos_combined)

rf_neg_tensor = torch.stack(tuple(rf_neg))
rf_neg_combined = torch.mean(rf_neg_tensor, dim=0)
rf_neg_array = np.array(rf_neg_combined)

rf_diff_array = rf_pos_array - rf_neg_array

# Re-orient the array for Heatmap visualization
flipped_array = np.transpose(rf_diff_array)
flipped_array = np.flip(flipped_array, 0)


# Create heatmap for the first dimension of the tensor
fig, ax = plt.subplots(figsize=(20, 10))
im = ax.imshow(flipped_array, aspect='auto')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Set axis labels and title
ax.set_xlabel('Residues + DSSP')  # Change this axis label if needed, current label is for OneHot+DSSP
ax.set_ylabel('Amino Acid Positions')
ax.set_title(f'Contrast {title_part} P-sites vs. non-P-sites')

# Add nicer tick labels to the axes
if 'onehot' in embed_type:
    xtick_labels = [aa for aa in 'ACDEFGHIKLMNPQRSTVWYUAIIIBCEH'] + ['n/a']
    counter = 0
    for i, aa in enumerate(xtick_labels):
        if aa == 'I':
            xtick_labels[i] = f'I{counter}' if counter > 0 else aa
            counter += 1
    xticks = [i for i in range(len(xtick_labels))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

ytick_labels = [str(-30 + (i * 5)) for i in range(13)][::-1]
yticks = [i * 5 for i in range(len(ytick_labels))]
ax.set_yticks(yticks)
ax.set_yticklabels(ytick_labels)

# Save the plot
path = f'/home/arendvc/{embed_abbrev.replace("+", "")}_{psite}_Model_Training_imgs/'
plt.savefig(path + f'Heatmap_{embed_abbrev}_{psite}_Difference.png')
