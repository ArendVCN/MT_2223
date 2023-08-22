import sys
from matplotlib import pyplot as plt
from Initiate_PSiteDataset import PSiteDataset, balance_dataset, re_init_dataset
from Model_analysis_functions import chance_performance_test
from AccessCorrectFolders import abbreviate_embed_names

'''
This script accepts the name of an embedding type (onehot, esm, carp, alphafold-t/-nt) and a P-site type (ST or Y)
and will run the test dataset on a trained model both normally and with shuffled labels a specified number of times 
in order to determine whether the observed AUC scores are statistically significant or by chance.

`python AUC_Significance.py <embed_type> <psite> <n_random_datasets>`
'''

# Load in the correct data files and (sub)directories
data_dir = "/data/home/arendvc/"
embed_type, psite, n_random_dls = sys.argv[1], sys.argv[2], int(sys.argv[3])
embed_path = embed_type.split('-')[0] + '_outputs'

if len(embed_type.split('-')) == 2:
    if embed_type.endswith('nt'):
        test_folder = "/test_NoTemp/"
    else:
        test_folder = "/test_Templated/"
else:
    test_folder = "/Ram22_test_windowed/"

embed_type = abbreviate_embed_names(embed_type)

embed_dir = data_dir + embed_path + test_folder
psite_file = "/home/arendvc/PSite_Files/Ram22_pep_filtered_" + psite + "_test.fasta"


# Create test dataset instances (both the original and randomly shuffled ones)
# To test the performance of the AUC's in relation to chance
test_data = PSiteDataset(data_rep_dir=embed_dir, labels_fasta=psite_file, field_radius=30)
balanced_test_data = balance_dataset(re_init_dataset(test_data))
model_path = data_dir + embed_path + f"/ModelV4_{psite}_CH40_rad30_{embed_type}.pth"

stat_dict = chance_performance_test(balanced_test_data, model_path, n_random_dls)
sample_pop_roc, obs_roc, t_stat_roc, p_value_roc = stat_dict['AUROC']
sample_pop_prc, obs_prc, t_stat_prc, p_value_prc = stat_dict['AURPC']


# Present the Chance Performance AUC results in a graph
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(10, 6))
fig.suptitle('Observed score vs. Sample of Random Permutations')

n1, bins1, patches1 = ax1.hist(sample_pop_roc, density=True, alpha=0.5)
ax1.axvline(obs_roc, color='red')
ax1.text(0.05, 0.95, f't = {t_stat_roc:.2f}\np = {p_value_roc:.4f}', transform=ax1.transAxes, va='top')

n2, bins2, patches2 = ax2.hist(sample_pop_prc, density=True, alpha=0.5)
ax2.axvline(obs_prc, color='red')
ax2.text(0.05, 0.95, f't = {t_stat_prc:.2f}\np = {p_value_prc:.4f}', transform=ax2.transAxes, va='top')

ax1.set_title('Distribution of AUROCs')
ax2.set_title('Distribution of AUPRCs')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
