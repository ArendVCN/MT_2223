import sys
import torch
from time import time
from torch import nn
from torch.utils.data import DataLoader
from Initiate_PSiteDataset import PSiteDataset, balance_dataset, re_init_dataset  # , get_subset_by_motif
from PSite_Models import PSitePredictV4
from Psite_Model_Training import train_model
from AccessCorrectFolders import access_folders, load_in_dicts
# from Model_analysis_functions import average_metricdicts, plot_model_hyperpar_changes

'''Instantiate objects of the PSite Dataset class for the given P-sites, with the given radius and dssp labels'''

embed_type, psite, radius, dssp, plot_arg = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
rep_dirs, lab_files = access_folders(embed_type, psite)


dssp_dict = None
if dssp == 'y':
    dssp_dict, _ = load_in_dicts(psite, "psite_amino_acids_numerical.tsv")


print("Creating Datasets...")
train_data = PSiteDataset(data_rep_dir=rep_dirs[0], labels_fasta=lab_files[0], field_radius=radius, dssp=dssp_dict)
test_data = PSiteDataset(data_rep_dir=rep_dirs[1], labels_fasta=lab_files[1], field_radius=radius, dssp=dssp_dict)
valid_data = PSiteDataset(data_rep_dir=rep_dirs[2], labels_fasta=lab_files[2], field_radius=radius, dssp=dssp_dict)


# print("Subsetting datasets:...")
# train_data = get_subset_by_motif(re_init_dataset(train_data))[1]
# test_data = get_subset_by_motif(re_init_dataset(test_data))[1]
# valid_data = get_subset_by_motif(re_init_dataset(valid_data))[1]

print("Balancing test & validation datasets...")
bal_test_data = balance_dataset(re_init_dataset(test_data))
bal_valid_data = balance_dataset(re_init_dataset(valid_data))

num_channels = len(train_data[0][0])  # Number of channels depends on the embedding

hyperparam_changes_dict = {}

start_time = time()
for rad in range(radius, radius - 1, -2):
    # if rad != radius:
    #     train_data.create_subwindow(rad)
    #     bal_test_data.create_subwindow(rad)
    #     bal_valid_data.create_subwindow(rad)

    '''Create the DataLoaders'''
    # batchsize = 64
    for batchsize in range(96, 193, 32):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        print("Radius:", rad)
        print(f"Creating DataLoaders (batchsize {batchsize})")
        train_dataloader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
        test_dataloader = DataLoader(dataset=bal_test_data, batch_size=batchsize, shuffle=False)
        valid_dataloader = DataLoader(dataset=bal_valid_data, batch_size=batchsize, shuffle=False)

        ####################################################################################################
        '''Create an instance of the model and train it'''

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        # Initiate an instance of the model
        for hidun in range(40, 41, 2):
            print("N° Hidden Units:", hidun)
            model = PSitePredictV4(input_shape=num_channels, hidden_units=hidun, output_shape=2,
                                   field_length=rad*2+1, kernel=3, pad_idx=1, dropout=0.3)
            model = model.to(device)
            print("Model created")

            # Define the loss function and optimizer
            weights = torch.tensor([1.0, 5.0])  # 1:5 works best for both, adjust when sub-setting the data
            loss_fn = nn.CrossEntropyLoss(weight=weights)  # More weight given to positive samples
            loss_fn = loss_fn.to(device)
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)

            # Train the model
            print("Training Model")
            trained_model = train_model(model, 25, train_dataloader, valid_dataloader, test_dataloader,
                                        loss_fn, optimizer, dev=device, plot=plot_arg)

            # hyperparam_changes.append((radius, trained_model[1]))
            # hyperparam_changes_dict.setdefault(k, []).append(trained_model[1])

end_time = time()

print("Total time passed:", round((end_time - start_time) / 3600, 2), "hours")

# hyperparam_changes = []
# for hp, bs_results_list in hyperparam_changes_dict.items():
#     hyperparam_changes.append((hp, average_metricdicts(bs_results_list)))
# plot_model_hyperpar_changes(hyperparam_changes)
