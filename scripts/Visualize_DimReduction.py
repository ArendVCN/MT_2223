import sys
import umap
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from Initiate_PSiteDataset import PSiteDataset, re_init_dataset, get_subset_by_motif
from AccessCorrectFolders import access_folders, load_in_dicts, abbreviate_embed_names

'''
Perform dimensionality reduction using PCA, t-SNE or UMAP and visualize them (+ the explained variance by the first
10 PC's). Visualization has only been applied to the validation dataset, since t-SNE and UMAP cannot seem to handle the
visualization of larger datasets.

Execute this script by calling: `python Visualize_DimReduction.py <embed_type> <P-site> <radius> <dr_type>
- embed_type = onehot, esm, carp, alphafold or the enhanced versions
- P-site = ST or Y
- radius = length of 1 side of the receptive field enclosing the central site (e.g. Radius 10 --> 10+S/T/Y+10 = RFL 21)
- dr_type = type of dimensionality reduction: pca, tsne or umap
'''

rec_fields = []
sites = []
y_labels = []
motifs = []
rsa_labels = []
ss3_labels = []
kinase_groups = []
proline = []

embed_type, psite, radius, dr_type = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
kinase_file = "KinasePsiteDictV2.json" if psite == 'ST' else None


rep_dirs, lab_files = access_folders(embed_type, psite)
dssp_dict, kinase_dict = load_in_dicts(psite, "psite_amino_acids_numerical.tsv", kinase_file)

embed_type = abbreviate_embed_names(embed_type)

# Create the datasets
valid_data = PSiteDataset(data_rep_dir=rep_dirs[2], labels_fasta=lab_files[2], field_radius=radius,
                          dssp=dssp_dict, kinase_specificity=kinase_dict)
# valid_data = get_subset_by_motif(re_init_dataset(valid_data))

if psite == 'ST':
    for X, y, seq, rsa, ss3, kinase in valid_data:
        if rsa != 'n/a':  # Leave out unusable data
            data = X.flatten()
            rec_fields.append(data)
            motifs.append(seq)
            sites.append(seq[len(seq) // 2])
            y_labels.append(y)
            rsa_labels.append(rsa)
            ss3_labels.append(ss3)

            if kinase in ('PIKK', 'PDHK', 'ALPHA', 'FAM20C'):
                kinase = 'Atypical'
            elif kinase == 'N/A':
                kinase = 'non-P-site'
            kinase_groups.append(kinase)

            if seq[len(seq) // 2 + 1] == 'P':
                proline.append(1)
            else:
                proline.append(0)

else:
    for X, y, seq, rsa, ss3 in valid_data:
        if rsa != 'n/a':
            data = X.flatten()
            rec_fields.append(data)
            y_labels.append(y)
            motifs.append(seq)
            rsa_labels.append(rsa)
            ss3_labels.append(ss3)


# Create a pandas dataframe from the tensors
rf_tensor = torch.stack(tuple(rec_fields))
rf_array = np.array(rf_tensor)
rf_array = StandardScaler().fit_transform(rf_array)

df = pd.DataFrame(rf_array)
df = df.assign(label=y_labels)
df = df.assign(rsa=rsa_labels)
df = df.assign(ss3=ss3_labels)
df = df.assign(motif=motifs)

if psite == 'ST':
    df = df.assign(kinase=kinase_groups)
    df = df.assign(site=sites)
    df = df.assign(proline=proline)
    df = df[df['kinase'] != 'Undefined']
    # df = df[df['proline'] == 0]  # Comment in or out for different views of the data
    # df = df[df['kinase'] != 'non-P-site']

# Restrict the RSA scores to discrete values by binning them
num_bins = 8
bin_labels = [f'[{i/num_bins:.2f}-{(i + 1)/num_bins:.2f}[' for i in range(num_bins)]
df['RSA-score'] = pd.cut(df['rsa'], bins=num_bins, labels=bin_labels)

print("DataFrame Constructed")

################################################################################################
''' Dimensionality reduction '''

label_colors_kin = None

if psite == 'ST':
    X = df.iloc[:, :-8]
    labels = df.iloc[:, -8:]

    # Define the colour palettes
    classes_kin = sorted(df['kinase'].unique())
    custom_palette_kin = sns.color_palette("muted", n_colors=len(classes_kin))
    cmgc = custom_palette_kin.pop(7)
    custom_palette_kin.insert(2, cmgc)
    ste = custom_palette_kin.pop(0)
    custom_palette_kin.insert(4, ste)
    combined = zip(sorted(classes_kin), custom_palette_kin)
    label_colors_kin = dict(combined)

    classes_lbl = sorted(df['label'].unique())
else:
    X = df.iloc[:, :-5]
    labels = df.iloc[:, -5:]
    classes_lbl = sorted(df['label'].unique())

custom_palette_lbl = sns.color_palette("muted", n_colors=len(classes_lbl))
combined = zip(sorted(classes_lbl), custom_palette_lbl)
label_colors_lbl = dict(combined)

classes_ss3 = sorted(df['ss3'].unique())
custom_palette_ss3 = sns.color_palette("Set1", n_colors=len(classes_ss3))
combined = zip(sorted(classes_ss3), custom_palette_ss3)
label_colors_ss3 = dict(combined)

# Generating plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))

if dr_type == 'pca':
    '''
    PCA
    '''
    print("Attempting PCA")
    pca = PCA(n_components=10)
    pca.fit(X)
    pca_data = pca.transform(X)

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)  # Calculates % of variation for each PC
    PC_labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

    # Plot explained variance for the 10 first PC's
    ax1.bar(x=range(1, len(per_var[:10]) + 1), height=per_var[:10], tick_label=PC_labels[:10])
    ax1.set_ylabel('Percentage of Explained Variance')
    ax1.set_xlabel('Principal Component')

    # Plot the first two PC's
    pca_df = pd.DataFrame(pca_data, columns=PC_labels, index=X.index)
    pca_df.sort_index(inplace=True)
    pca_df1 = pd.concat([pca_df, labels], axis=1)

    if psite == 'ST':
        sns.scatterplot(x=pca_df1.PC1, y=pca_df1.PC2, palette=label_colors_kin, data=pca_df1, hue='kinase', style='site', ax=ax2, s=20)
    else:
        sns.scatterplot(x=pca_df1.PC1, y=pca_df1.PC2, palette=label_colors_lbl, data=pca_df1, hue='label', ax=ax2, s=15)
    ax2.set_xlabel(f'PC1 - {per_var[0]:.2f}%')
    ax2.set_ylabel(f'PC2 - {per_var[1]:.2f}%')

    sns.scatterplot(x=pca_df1.PC1, y=pca_df1.PC2, palette='viridis', data=pca_df1, hue='RSA-score', ax=ax3, s=15)
    ax3.set_xlabel(f'PC1 - {per_var[0]:.2f}%')
    ax3.set_ylabel(f'PC2 - {per_var[1]:.2f}%')

    sns.scatterplot(x=pca_df1.PC1, y=pca_df1.PC2, palette=label_colors_ss3, data=pca_df1, hue='ss3', ax=ax4, s=15)
    ax4.set_xlabel(f'PC1 - {per_var[0]:.2f}%')
    ax4.set_ylabel(f'PC2 - {per_var[1]:.2f}%')

    fig.suptitle(f"{embed_type} {psite} Data - Dimensionality reduction (PCA)")

elif dr_type == 'tsne':
    '''
    t-SNE
    '''
    print("Attempting t-SNE")

    perpl = 75
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perpl,
                random_state=42, method='exact', n_iter=500)
    tsne_results = tsne.fit_transform(X)
    tsne_df = pd.DataFrame({'tSNE_1': tsne_results[:, 0], 'tSNE_2': tsne_results[:, 1]}, index=X.index)
    tsne_df1 = pd.concat([tsne_df, labels], axis=1)

    lim1 = (tsne_results.min()-0.5, tsne_results.max()+0.5)

    sns.scatterplot(x='tSNE_1', y='tSNE_2', hue='label', palette=label_colors_lbl, data=tsne_df1, ax=ax1, s=15)
    ax1.set_xlim(lim1)
    ax1.set_ylim(lim1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    sns.scatterplot(x='tSNE_1', y='tSNE_2', hue='RSA-score', palette='viridis', data=tsne_df1, ax=ax2, s=15)
    ax2.set_xlim(lim1)
    ax2.set_ylim(lim1)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    sns.scatterplot(x='tSNE_1', y='tSNE_2', hue='ss3', palette=label_colors_ss3, data=tsne_df1, ax=ax3, s=15)
    ax3.set_xlim(lim1)
    ax3.set_ylim(lim1)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)

    if label_colors_kin:
        sns.scatterplot(x='tSNE_1', y='tSNE_2', hue='kinase', palette=label_colors_kin, data=tsne_df1, ax=ax4, s=15)
        ax4.set_xlim(lim1)
        ax4.set_ylim(lim1)
        ax4.axes.get_xaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)

    fig.suptitle(f"{embed_type} {psite} Data - Dimensionality reduction (t-SNE - perplexity {perpl})")

else:
    '''
    UMAP
    '''
    print("Attempting UMAP")

    neighbors = 20
    reducer = umap.UMAP(n_neighbors=neighbors, n_components=2, min_dist=0.1,
                        random_state=42, metric='euclidean')
    umap_results = reducer.fit_transform(X)
    umap_df = pd.DataFrame({'UMAP_1': umap_results[:, 0], 'UMAP_2': umap_results[:, 1]}, index=X.index)
    umap_df1 = pd.concat([umap_df, labels], axis=1)

    lim2 = (umap_results.min()-0.5, umap_results.max()+0.5)

    # Uncomment this in order to label the data points with the amino acid sequences
    # idc = umap_df1.index.tolist()
    # x_val, y_val, txt_val = [], [], []
    # for i in idc:
    #     x_val.append(umap_df1['UMAP_1'][i])
    #     y_val.append(umap_df1['UMAP_2'][i])
    #     txt_val.append(umap_df1['motif'][i])

    sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='site', data=umap_df1, ax=ax1, s=15)
    ax1.set_xlim(lim2)
    ax1.set_ylim(lim2)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='RSA-score', palette='viridis', data=umap_df1, ax=ax2, s=15)
    ax2.set_xlim(lim2)
    ax2.set_ylim(lim2)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='ss3', palette=label_colors_ss3, data=umap_df1, ax=ax3, s=15)
    ax3.set_xlim(lim2)
    ax3.set_ylim(lim2)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)

    if label_colors_kin:
        sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='kinase', palette=label_colors_kin, data=umap_df1, ax=ax4, s=15)
        ax4.set_xlim(lim2)
        ax4.set_ylim(lim2)
        ax4.axes.get_xaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)

    # Uncomment this as well for sequence labelling
    # for i in range(len(umap_df1)):
    #     ax1.text(x_val[i], y_val[i], txt_val[i], fontsize=10, ha='center', va='center')
    #     ax2.text(x_val[i], y_val[i], txt_val[i], fontsize=10, ha='center', va='center')
    #     ax3.text(x_val[i], y_val[i], txt_val[i], fontsize=10, ha='center', va='center')
    #     ax4.text(x_val[i], y_val[i], txt_val[i], fontsize=10, ha='center', va='center')

    fig.suptitle(f"{embed_type} {psite} Data - Dimensionality reduction (UMAP - neighbors {neighbors})")

fig.tight_layout(h_pad=10, w_pad=5)
plt.subplots_adjust(top=0.95)

plt.savefig(f'/home/arendvc/DimRed_{embed_type}_{psite}_{radius}_Cl1_Flattened_AllSites.png')
# plt.show()
