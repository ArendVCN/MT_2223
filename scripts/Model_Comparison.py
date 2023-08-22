import sys
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import Paired
from torch.utils.data import DataLoader
from AccessCorrectFolders import abbreviate_embed_names
from Initiate_PSiteDataset import PSiteDataset, re_init_dataset, balance_dataset
from PSite_Models import PSitePredictV4
from Model_analysis_functions import test_model, average_output_tuples, calculate_performance_metrics

'''
Compares the performance of saved PSitePredict prediction models trained on differently-derived embeddings
(OneHot, ESM, CARP, AlphaFold) by testing on the same dataset and graphing the ROC and PRC Areas Under the Curve.

Run this script from the command line by giving the additional argument 'ST' or 'Y' to select which P-sites to test.
'''

print("Loading in Necessary data...")
data_dir = "/data/home/arendvc/"

# Choose the embeddings for which the models will be tested and select the correct source directories of the test data:
embeddings = ['onehot', 'esm', 'carp', 'alphafold-t', 'alphafold-nt']
embeddings = embeddings + list(map(lambda x: x + '+DSSP', embeddings))

title_embs = list(map(abbreviate_embed_names, embeddings))
title_order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
title_embs = [title_embs[i] for i in title_order]

embed_dirs = [data_dir + embed.replace('-nt', '').replace('-t', '') + '_outputs/' for embed in embeddings]
embed_dirs = [embed_dirs[i] for i in title_order]

test_dirs = ['None'] * len(embed_dirs)
for n, edir in enumerate(embed_dirs):
    if n in [6, 7]:
        test_dirs[n] = edir + 'test_Templated/'
    elif n in [8, 9]:
        test_dirs[n] = edir + 'test_NoTemp/'
    else:
        test_dirs[n] = edir + 'Ram22_test_windowed/'

psite = sys.argv[1]
test_file = "/home/arendvc/PSite_Files/Ram22_pep_filtered_" + psite + "_test.fasta"


print("Creating Test Datasets...")
test_dls = []
for m, tdir in enumerate(test_dirs):
    ds = PSiteDataset(data_rep_dir=tdir, labels_fasta=test_file, field_radius=30)
    ds = re_init_dataset(ds)
    # bal_ds = balance_dataset(ds)

    channel_nr = len(ds[0][0])

    replicates = []
    for bs in range(64, 193, 32):
        dl = DataLoader(dataset=ds, batch_size=bs, shuffle=True)
        replicates.append(dl)

    test_dls.append((replicates, channel_nr))

# Load in the saved models and run each model on the respective test data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fill in the appropriate models
print("Loading in Saved Models...")
model_paths = []
for embed_type, embed_dir in zip(title_embs, embed_dirs):
    model_paths.append(embed_dir + f'ModelV4_{psite}_CH40_rad30_{embed_type}_Cl2.pth')


outputs = []
for n, (test_dl, num_features) in enumerate(test_dls):
    model = PSitePredictV4(input_shape=num_features,
                           hidden_units=40,
                           output_shape=2,
                           field_length=61,
                           kernel=3,
                           pad_idx=1,
                           dropout=0.3)

    loaded_state_dict = torch.load(model_paths[n])
    model.load_state_dict(loaded_state_dict)
    model = model.to(device)

    print(f"Testing model {n + 1}...")

    model_reps = []
    for rep in test_dl:
        preds = test_model(model, rep, device)
        model_reps.append(calculate_performance_metrics(*preds))
    model_avg = average_output_tuples(model_reps)

    outputs.append(model_avg)


# Changes to make the graph nicer
title_label = 'Ser/Thr' if psite == 'ST' else 'Tyr'
cols = list(Paired.colors)

# Plot the outcomes of the different models in a comparative graph
plt.figure(figsize=(18, 8))

# Plot ROC curve
print("Generating Plots...")
plt.subplot(1, 2, 1)
for m, output in enumerate(outputs):
    plt.plot(output[3], output[4], label=f'{title_embs[m]} ({output[0]:.3f})', color=cols[m])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right", title="Encoder (AUC)")

# Plot Precision-Recall curve
plt.subplot(1, 2, 2)
for m, output in enumerate(outputs):
    plt.plot(output[5], output[6], label=f'{output[1]:.3f} ({output[2]:.3f})', color=cols[m])

    # Also prints the mean accuracy scores for each model
    print(title_embs[m], f"Total mean average: {output[7]*100:.2f}")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall curve')
plt.legend(loc="lower left", title="AUC (F1-score)")

plt.suptitle(f"PSitePredict Performance Comparison - {title_label}")
plt.savefig(data_dir + f'Model4_AUC_Comparison_CH40_R30_{psite}_5runs.png')
