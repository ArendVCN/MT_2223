import sys
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import Paired
from torch.utils.data import DataLoader
from PSite_Models import PSitePredictV4
from Initiate_PSiteDataset import PSiteDataset, balance_dataset, re_init_dataset
from Model_analysis_functions import bootstrapping, test_model, randomized_permutation_test

'''
Run from the command line.

This script accepts the name of two embedding types (onehot, esm, carp, alphafold) and a P-site type (ST or Y)
and will run a test dataset on both models. It will then calculate whether there is a statistically significant change 
between them or not. 
A fourth argument allows to choose between bootstrap and permutation shuffle analysis
A fifth argument allows to choose a random seed for the randomized permutations

`python ModelCurve_Significance.py <embed_type1> <embed_type2> <psite> <test_type> <n_samples> <random_state>`
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load in the correct data files and directories
data_dir = "/data/home/arendvc/"

embed_types, psite, test = [sys.argv[1], sys.argv[2]], sys.argv[3], sys.argv[4]
n_samples, random_state = int(sys.argv[5]), int(sys.argv[6])

if test not in ('perm', 'boot'):
    raise KeyError("Test must be either set to 'boot' or 'perm'")

psite_file = "/home/arendvc/PSite_Files/Ram22_pep_filtered_" + psite + "_test.fasta"
title_label = 'Ser/Thr' if psite == 'ST' else 'Tyr'

test_folders, embed_folders = [], []

for embed_type in embed_types:
    if 'AF' in embed_type:
        test_folder = "/test_NoTemp/" if 'n' in embed_type else "/test_Templated/"
    else:
        test_folder = "/Ram22_test_windowed/"

    test_folders.append(test_folder)

    embed_path = embed_type.lower().replace('oh', 'onehot').replace('aft', 'alphafold').replace('afnt', 'alphafold').replace('+', '+DSSP') + '_outputs'
    embed_folders.append(embed_path)


aurocs, auprcs = [], []
models, true = [], None

# Create the saved model paths and test datasets
for i, embed_path in enumerate(embed_folders):
    model_path = data_dir + embed_path + f"/ModelV4_{psite}_CH40_rad30_{embed_types[i]}.pth"
    embed_dir = data_dir + embed_path + test_folders[i]

    # Create test dataset
    test_data = PSiteDataset(data_rep_dir=embed_dir, labels_fasta=psite_file, field_radius=30)
    test_data = balance_dataset(re_init_dataset(test_data))  # (Un)comment if you like normal or balanced datasets
    channels = len(test_data[0][0])

    # Create test dataloaders
    dl = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    # Instantiate the model
    model = PSitePredictV4(input_shape=channels,
                           hidden_units=40,
                           output_shape=2,
                           field_length=61,
                           kernel=3,
                           pad_idx=1,
                           dropout=0.3)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # Apply test dataset
    y_probs, _, y_true = test_model(model, dl, device)

    models.append(y_probs)
    true = y_true


# Choose which analysis to perform: bootstrapping or random permutation
if test == 'perm':
    counter_roc, counter_prc = 0, 0

    auc_1, auc_2, auroc_3, auprc_3 = randomized_permutation_test(models[0], models[1], true, n_samples, random_state)

    # Calculate the number of times a randomly permuted dataset performs better than one of the original datasets
    for i, score in enumerate(auroc_3):
        if score > auc_2[0]:
            counter_roc += 1
        if auprc_3[i] > auc_2[1]:
            counter_prc += 1

    # Divided by the total number of permutations results in a p-value
    p_roc, p_prc = counter_roc / n_samples, counter_prc / n_samples

    print(f"Significance level between both ROC: {p_roc:e}\nSignificance level between both PRC: {p_prc:e}")

else:
    model_scores = bootstrapping(models[0], models[1], true, n_samples, random_state)

    for m in model_scores:
        aurocs.append(m[:2])
        auprcs.append(m[2:])

    # Plot histograms of the AUC Bootstrapping distributions
    cols = list(Paired.colors)

    plt.figure(figsize=(12, 14))
    plt.subplot(2, 1, 1)
    for s, auroc in enumerate(aurocs):
        plt.hist(auroc[0], label=f'{embed_types[s]}: {auroc[1]}', alpha=0.4, color=cols[(s*2)+1], bins=10)
    plt.xlim([0.5, 1.0])
    plt.title(f'AUROC Score Distribution')
    plt.legend(loc="upper left", title="CI (95%)")

    plt.subplot(2, 1, 2)
    for s, auprc in enumerate(auprcs):
        plt.hist(auprc[0], label=f'{embed_types[s]}: {auprc[1]}', alpha=0.4, color=cols[(s*2)+1], bins=10)
    plt.xlim([0.5, 1.0])
    plt.xlabel('Distribution')
    plt.title(f'AUPRC Score Distribution')
    plt.legend(loc="upper left", title="CI (95%)")

    plt.suptitle(f"AUC Bootstrapping - {embed_types[0]} vs. {embed_types[1]} ({title_label})")
    plt.savefig(data_dir + f'BootstrapLinked_AUCScore_{psite}_{embed_types[0]}vs{embed_types[1]}_{n_samples}({random_state}).png')
