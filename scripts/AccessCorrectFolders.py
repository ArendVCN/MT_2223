import os
import json
from Initiate_PSiteDataset import make_dssp_dict

'''
This script contains functions that automatically access the correct file and directory paths for loading in 
the necessary data. Functions in this script:
- access_folders(): access the directories with embedded protein data and labelled protein fasta files
- abbreviate_embed_names(): shortens the names of embedding types (e.g. alphafold) to something more usable (e.g. AF)
- load_in_dicts(): load in dictionaries with structural data and Kinase group data (optional, only for Ser/Thr data)
'''


def access_folders(embed_type: str, psite: str):
    """
    Function automatically selects the correct path locations for the files with ML embedded protein data and the
    fasta files containing the proteins with labelled phosphorylation sites for the train, test and validation datasets.

    Output consists of 2 lists containing (1) the train-, test- and validation embeddings, and (2) the train-, test-
    and validation labelled fasta files.

    - embed-type (str): onehot, esm, carp, alphafold-nt, alphafold-t or the enhanced versions (add +DSSP)
    - psite (str): ST or Y
    """
    af_suffix = None

    # AlphaFold with and without templates is stored in separate subdirectories, identified by the suffix `nt` or `t`
    if embed_type.startswith('alphafold'):
        embed_type, af_suffix = embed_type.split('-')

    # Write the paths to the train, test and valid datasets
    rep_home_folder = "/data/home/arendvc/"
    rep_path = os.path.join(rep_home_folder, embed_type)

    datasets = ["train", "test", "valid"]

    # Identify the embeddings' destination folders for the train, test and validation datasets
    if af_suffix:
        if af_suffix.startswith('nt'):
            rep_dirs = [rep_path + "/" + ds + "_NoTemp" for ds in datasets]
        else:
            rep_dirs = [rep_path + "/" + ds + "_Templated" for ds in datasets]
    else:
        rep_dirs = [rep_path + "/Ram22_" + ds + "_windowed" for ds in datasets]

    # Identify the home folders of the labelled FASTA files for the three datasets
    lab_dir_name = "/home/arendvc/PSite_Files"
    lab_files = [os.path.join(lab_dir_name, "Ram22_pep_filtered_" + psite + "_" + ds + ".fasta") for ds in datasets]

    return rep_dirs, lab_files


def abbreviate_embed_names(embedding: str):
    """
    Abbreviates the name of the embedding type to make it easier to use in graphs/plots and file names.
    """
    embedding_list = embedding.replace('DSSP', '').replace('onehot', 'oh').replace('alphafold', 'af').split('-')

    if len(embedding_list) == 1:
        embed_type = embedding_list[0].upper()
    else:
        embed_type = embedding_list[0].upper() + embedding_list[1]

    return embed_type


def load_in_dicts(psite: str, dssp_file: str, kinase_file: str = None):
    """
    Loads in 1 (DSSP) or more (DSSP & Kinase family) dictionaries to add to the dataset creation to help in further
    analyses. Needs specification of the P-site, as DSSP dicts are available for both ST and Y data, but Kinase groups
    only apply to ST data.

    Returns the dssp_dict created by Initiate_PSiteDataset.make_dssp_dict, optionally kinase_dict saved to a .json file.

    - psite (str): ST or Y
    - dssp_file: .tsv file containing position-specific DSSP data
    - kinase_file: .json file containing a dictionary with the major binding kinase per P-site per protein
    """
    home_path = '/home/arendvc/'
    dssp_path, kin_path = home_path + 'dssp/', home_path + 'Kinase_Specificity/'

    assert not (psite != 'ST' and kinase_file is not None), "Kinase groups are only specified for Ser and Thr"

    # Create DSSP dict
    dssp_dict = make_dssp_dict(dssp_path + dssp_file, psite)

    # Load in Kinase dict if given
    if kinase_file is not None:
        with open(kin_path + kinase_file, 'r') as f:
            kinase_dict = json.load(f)
    else:
        kinase_dict = None

    return dssp_dict, kinase_dict
