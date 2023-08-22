import os
import torch
import torch.nn.functional as func
from Initiate_PSiteDataset import PSiteDataset, ReInitPSiteDataset


def add_dssp2pos(dssp_dict: dict, prot_id: str, pos: int):
    """
    Secondary structure information of an amino acid with position `pos` in protein `prot_id` is returned as
    a one-hot encoded tensor, depending on the information in the presented DSSP dictionary.
    Residue solvent accessible surface is divided into 5 categories:
    A: fully accessible,
    I1-3: 3 different stages of interface, and
    B: buried.
    An additional column is added to the onehot encoding in case no secondary structure information is known.

    Returns a torch tensor with shape 1x9.
    """
    sec_str = []
    labels = {'A': 0, 'I1': 1, 'I2': 2,
              'I3': 3, 'B': 4, 'C': 5,
              'E': 6, 'H': 7, 'n/a': 8}

    for aa in dssp_dict[prot_id]:
        if aa[0] == pos:
            sec_str.append(aa[3])
            if aa[4] < 0.2:
                sec_str.append('B')
            elif aa[4] < 0.4:
                sec_str.append('I3')
            elif aa[4] < 0.6:
                sec_str.append('I2')
            elif aa[4] < 0.8:
                sec_str.append('I1')
            else:
                sec_str.append('A')

    if not sec_str:
        sec_str.append('n/a')

    output = torch.tensor([labels[i] for i in sec_str])
    output = func.one_hot(output, num_classes=len(labels))

    # If the amino acid has known values, it will result in a 2x9 tensor, which needs to be condensed into a 1x9 tensor
    if len(output) == 2:
        output = torch.sum(output, dim=0)
    output.to('cuda' if torch.cuda.is_available() else 'cpu')

    return output.squeeze()


def add_dssp2recfield(rep: torch.Tensor, prot_id: str, psite_pos: int, dssp_dict: dict):
    """
    Adds structural information to the entire receptive field of an embedded representation by adding the tensors
    created by add_dssp2pos() to each position in the embedding.

    Input:
    - the original tensor,
    - the protein ID,
    - the position of the central site of the receptive field,
    - a dictionary with structural (DSSP) information.

    Returns the modified representation.
    """
    rep_rev = rep.permute(1, 0)  # Rep is now (Length, Channel) instead of (Channel, Length)

    # Create the 1x9 one-hot structure data tensors for each position
    radius = len(rep_rev) // 2
    positions = list(range(psite_pos - radius, psite_pos + radius + 1))
    pos_dssp = [add_dssp2pos(dssp_dict, prot_id, pos) for pos in positions]

    # Concatenate the structure tensors to the original tensors for each position in the receptive field
    mod_pos_reps = []
    for i, pos_rep in enumerate(rep_rev):
        mod_pos_reps.append(torch.cat((pos_rep, pos_dssp[i]), dim=0))

    # Stack the separate 1xC+9 tensors into one multidimensional LxC+9 tensor
    mod_pos_reps = torch.stack(mod_pos_reps, dim=0)

    output = mod_pos_reps.permute(1, 0)  # Convert tensor back to (Channel+9, Length)
    return output


def add_dssp2rep(rep: torch.Tensor, prot_id: str, dssp_dict: dict):
    """
    Adds structural information to the entire embedded representation by adding the tensors created
    by add_dssp2pos() to each position in the embedding.

    Input:
    - the original tensor,
    - the protein ID,
    - a dictionary with structural (DSSP) information.

    Returns the modified representation.
    """
    pos_dssp = [add_dssp2pos(dssp_dict, prot_id, pos) for pos in range(len(rep))]

    mod_pos_reps = []
    for i, pos_rep in enumerate(rep):
        mod_pos_reps.append(torch.cat((pos_rep, pos_dssp[i]), dim=0))

    output = torch.stack(mod_pos_reps, dim=0)

    return output


def add_dssp2dataset(dataset: PSiteDataset, dssp_dict: dict):
    """
    Adds structural information to all representations within a PSiteDataset.

    Input:
    - a PSiteDataset instance,
    - a dictionary with structural (DSSP) information.

    Returns an instance of a ReInitDataset, with the modified representations.
    """
    data = []
    labels = []
    seqs = []
    extra1 = None
    extra2 = None
    extra3 = None

    # Identify whether and which structural information was added to the dataset
    # Convert the appropriate NoneTypes to an empty list
    n_variables = len(dataset[0])
    if n_variables > 3:
        if n_variables >= 5:
            extra1, extra2 = [], []
            if n_variables > 5:
                extra3 = []
        else:
            extra3 = []

    # For each sample in the dataset, modify the representations through add_dssp2recfield()
    for i, (X, y, seq, *extra) in enumerate(dataset):
        pept_id, prot_pos, *_ = dataset.psite_pos[i]
        prot_id = pept_id.split('_')[0] if '_' in pept_id else pept_id

        new_X = add_dssp2recfield(X, prot_id, prot_pos - 1, dssp_dict)

        data.append(new_X)
        labels.append(y)
        seqs.append(seq)

        if len(extra) > 0:
            if len(extra) >= 2:
                extra1.append(extra[0])
                extra2.append(extra[1])
                if len(extra) > 2:
                    extra3.append(extra[2])
            else:
                extra3.append(extra[2])

    output = ReInitPSiteDataset(rf_embeddings=data,
                                labels=labels,
                                rf_sequences=seqs,
                                rsa=extra1,
                                ss3=extra2,
                                kinase=extra3)

    print('DSSP added to Dataset')
    return output


def add_dssp2datadir(dir_loc: str, dssp_dict: dir):
    """
    Adds extra channels with structural information to the tensor representation within every .pt file within the given
    directory. Is similar to add_dssp2dataset, but does it to the entire representation instead of the receptive field,
    while also permanently storing the resulting representations into a new directory. Takes less time to compute for
    larger representations.

    Input:
    - home directory of the original representations (onehot, esm, carp or alphafold-t/-nt
    - a dictionary with structural (DSSP) information.

    Returns a new directory containing .pt files with the modified representations. The new directory has the same name
    as the original directory with the suffix '+DSSP'.
    """
    path = '/data/home/arendvc/'

    # Generate the subdirectories corresponding to all three datasets
    datasets = ('train', 'test', 'valid')
    if 'alphafold' in dir_loc:
        if dir_loc.endswith('nt'):
            dir_locs = [path + dir_loc + f'_outputs/{ds}_NoTemp/' for ds in datasets]
        else:
            dir_locs = [path + dir_loc + f'_outputs/{ds}_Templated/' for ds in datasets]
    else:
        dir_locs = [path + dir_loc + f'_outputs/Ram22_{ds}_windowed/' for ds in datasets]

    # Create the new directories where the modified data will be located
    for d in dir_locs:
        print(d)
        new_embed_dir = path + dir_loc + '+DSSP_outputs'
        if not os.path.exists(new_embed_dir):
            os.mkdir(new_embed_dir)
        new_embed_dir_sub = new_embed_dir + '/' + d.split('/')[-2]
        print(new_embed_dir_sub)
        os.mkdir(new_embed_dir_sub)

        not_in_dir = 0
        for file in os.listdir(d):
            outfile = new_embed_dir_sub + '/' + file
            if file.endswith('.pt') and not os.path.exists(outfile):
                # Create new .pt files with the same names as the original files
                prot_id = file.split('.')[0].split('_')[0]

                # Does not create new .pt files for proteins for which no structural data was available
                try:
                    key_exists = dssp_dict[prot_id]
                    with open(d + file, "rb") as f:
                        rep = torch.load(f, map_location=torch.device('cpu'))
                    new_rep = add_dssp2rep(rep, prot_id, dssp_dict)
                except KeyError:
                    not_in_dir += 1
                    continue

                torch.save(new_rep, outfile)

        print('Files not modified:', not_in_dir)
