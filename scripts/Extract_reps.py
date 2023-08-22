import os
import pickle
import torch
import numpy as np

'''
This script extracts the specified models from the 5 alphafold output models (here: models 1 & 3) and constructs new
directories which can be accessed by other scripts for feature extraction/model training. Additionally, it converts
the pickled files (.pkl) created by AlphaFold to PyTorch files (.pt), so there is consistency with the other algorithms.

Some AlphaFold models were pre-trained on structural templates (1-2), while other were not (3-5). Consequebntly, I
elected to extract one of both.
'''


path = '/data/home/arendvc/alphafold_outputs/'
datasets = ['train', 'test', 'valid']
indirs = ['Ram22_' + ds + '_windowed/' for ds in datasets]
outdirs = ['Templated/', 'NoTemp/']

# Create the output directories for the extracted models
for dataset in datasets:
    for out in outdirs:
        outpath = path + dataset + '_' + out
        if not os.path.exists(outpath):
            os.mkdir(outpath)

for indir in indirs:
    print("Currently in Dir:", indir)
    for d in os.listdir(path + indir):
        prot_id = str(d) + '.pt'
        loc = os.path.join(path + indir, d)

        # Extract the first and third embedded representations (which are stored in pickled files)
        reps = []
        for i in [1, 3]:
            with open(loc + f'/repr_single_model_{i}_ptm.pkl', 'rb') as f:
                reps.append(pickle.load(f))

        # Transform the representations into tensors
        for i, rep in enumerate(reps):
            rep = torch.from_numpy(np.array(rep))

            # Save the rep as a file in the correct folder
            torch.save(rep, path + indir.split('_')[1] + '_' + outdirs[i] + prot_id)
