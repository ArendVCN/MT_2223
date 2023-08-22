import os
import csv
import random
import torch
from torch.utils.data import Dataset
from math import ceil
from Bio import SeqIO

'''
Custom dataset created that takes a directory with the protein/peptide representations derived from a ML algorithm
+ a FASTA file with the full protein sequences labelled for putative P-sites
+ the classes used to label P-sites as a dictionary (e.g. {label1: 0, label2: 1, ... labeln: n-1}
+ an arbitrary radius of the receptive field (e.g. radius 15 --> receptive field from [pos-15:pos+15] (right-included))
'''

####################################################################################################
'''Defining functions for use in the class creation'''


def make_dssp_dict(dssp_psite_file, ptype: str = 'ST'):
    """
    Creates a dictionary instance of a tab-delimited file containing secondary protein structure information
    about the putative P-sites. The type of P-site can be specified as ST or Y (default = ST).
    """
    dssp_dict = {}

    with open(dssp_psite_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            # Select the positions with a serine/threonine or tyrosine
            if row[2] in ptype:
                # Dict key = protein ID, values = tuple with location + structural information
                dssp_dict.setdefault(row[0], []).append((int(row[1]), row[3], row[4], row[5], float(row[6])))

    return dssp_dict


def find_sec_str(dssp_dict: dict, prot_id: str, pos: int, output: str = 'rsa'):
    """
    Using the given dictionary made by `make_dssp_dict`, finds the secondary structure information of the given
    amino acid position in a specific protein. Depending on the value for `output`, returns the residue solvent
    accessibility score (= rsa) or the three-state secondary structure denomination (C: loop, E: strand or H: helix)
    If no match is found in the dictionary, 'n/a' is returned.

    - dssp_dict: dictionary output from make_dssp_dict
    - prot_id: protein accession ID
    - pos: residue position in the protein sequence (starting from 0)
    - output: 'ss3' or 'rsa' for the secondary structure label and solvent accessibility score, respectively
    """
    assert output in ('rsa', 'ss3'), "Output cannot be anything other than 'rsa' or 'ss3'"

    for aa in dssp_dict[prot_id]:
        if int(aa[0]) == pos:
            if output == 'rsa':
                return aa[4]
            else:
                return aa[3]

    return 'n/a'


def find_kinase(kinase_dict: dict, prot_id: str, pos: int, psite_status: int):
    """
    Using a stored dictionary (KinasePsiteDictV2.json), finds the appropriate Kinase protein family
    for the given protein ID and P-site position. Only goes for ST P-sites. If the data is not present
    in the dictionary, returns 'Undefined' if psite_status is positive and 'N/A' if negative.
    """
    if prot_id not in kinase_dict:
        return 'Undefined' if psite_status == 1 else 'N/A'

    for aa in kinase_dict[prot_id]:
        if aa[0] == pos:
            return aa[1]

    return 'Undefined' if psite_status == 1 else 'N/A'


def create_pept_dict(pkl_dir):
    """
    Input is a directory containing pytorch files containing the peptide representations.
    Output is a dictionary with the peptide name and embedding as key:value pair.
    The embeddings consist of two dimensions: (1) the length of the peptide and (2) the number of channels.
    It is assumed the pytorch file names are the peptide IDs.
    """
    tensor_dict = {}

    for loc in os.scandir(pkl_dir):
        # Catch potential errors should a subdirectory be present in the given directory
        try:
            assert loc.name.split(".")[-1] == "pt", f"{loc} NOT a PYTORCH (.pt) file"
            fh = open(loc, "rb")
            pept_id = loc.name.split(".")[0]
            rep = torch.load(fh, map_location=torch.device('cpu'))
            # rep = torch.squeeze(rep)  # Remove unnecessary dimension (i.e. the batch dimension if applicable)
            tensor_dict[pept_id] = rep

        except IsADirectoryError:
            # Ignore it if there are subdirectories within the current folder
            pass

    return tensor_dict


def define_psites(labelled_fasta_file):
    """
    The sequences in the input file (.fasta) have been labeled with special characters immediately following
    the putative PTM site for each protein sequence. Since some proteins are too large to be processed, they are split
    into peptides of maximally 900 AA, with 450 AA overlaps. These peptides are labelled as <prot_id>_<integer>.
    This function returns the peptide ID, numerical position of the site in the protein sequence AND in the peptide
    sequence, and the modification label (1: positive or 0: negative).
    """
    classes = {'@': 0, '#': 1}

    psites = []  # as [(pept_id, prot_position, pept_position, label),...]
    prot_seqs = {}  # as {prot_id: sequence}

    for seq_record in SeqIO.parse(labelled_fasta_file, "fasta"):
        prot_id, prot_seq = seq_record.id, str(seq_record.seq)

        # Find the length of the protein
        unlabelled_prot_seq = prot_seq.replace("@", "").replace("#", "")
        prot_len = len(unlabelled_prot_seq)

        # Store the full protein sequences
        prot_seqs[prot_id] = unlabelled_prot_seq

        # Define number of peptides to divide this protein in
        pept_n = ceil((prot_len - 900) / 450) + 1

        # Capture the offsets used to select the correct peptide to which to assign the PTM site
        offsets = []
        for i in range(pept_n):
            offsets.append(675 + 450 * i)  # 900 AA peptides + 450 overlap: offset 1 at 675, others at 450 intervals

        aa_prot = 0  # Keep note of the amino acid position
        for char in prot_seq:
            # Determine the peptide ID when a P-site is found
            if char in classes.keys():
                # If protein is split into peptides, find the correct peptide based on the offset
                if pept_n > 1:
                    suffix = 0
                    found = False

                    # This chunk determines to which peptide a P-site will be assigned
                    while not found and not suffix == pept_n:  # Terminate when peptide is found OR search exhausted
                        suffix += 1
                        if aa_prot <= offsets[suffix - 1]:
                            found = True

                    # If the final peptide is too small, extend the overlap
                    if suffix == pept_n and prot_len - 450 * (suffix - 1) <= 600:
                        aa_pept = aa_prot - (prot_len - 600)
                    else:
                        aa_pept = aa_prot - 450 * (suffix - 1)

                    pept_id = prot_id + f"_{suffix}"

                # If protein has not been split in peptides, just use protein ID as peptide ID
                else:
                    aa_pept = aa_prot
                    pept_id = prot_id

                psites.append((pept_id, aa_prot, aa_pept, classes[char]))

            else:
                aa_prot += 1

    return psites, prot_seqs


####################################################################################################
'''Create a class for generating custom datasets'''


class PSiteDataset(Dataset):
    def __init__(self, data_rep_dir, labels_fasta, field_radius: int,
                 dssp: dict = None, kinase_specificity: dict = None):
        super().__init__()
        self.radius = field_radius
        self.data = create_pept_dict(data_rep_dir)  # Extract the dataset peptide embeddings and convert to dictionary
        self.psite_pos, self._protseqs = define_psites(labels_fasta)  # the P-sites sites are extracted and stored
        self.__assert_correct_init()  # Assert whether the input has been correctly handled
        self.dssp_dict = dssp
        self.kinase_spec = kinase_specificity
        print("Dataset created")

    def __assert_correct_init(self):
        """
        Assert whether the input data from the data_rep_directory has the same peptide IDs
        as the P-site labels derived from the original fasta file. Throws an AssertionError if not.
        The number of peptide IDs in `self.data` must be either equal to or greater than those in `self.labels`,
        since some peptides don't have any assigned sites due to the way `self.labels` is formatted.
        """
        set1 = set(pid[0] for pid in self.psite_pos)
        set2 = set(self.data.keys())

        assert set1.issubset(set2), "Dataset Error: the following peptides are missing from the Embeddings:\n" \
                                    f"{sorted(set2 - set1)}"

    def __len__(self):
        return len(self.psite_pos)

    def __getitem__(self, idx):
        """
        The method is rewritten to take into account that the stored P-site position in the full protein is not the same
        as the one stored in the peptide representation.
        A receptive field around the P-site is selected with a length depending on the input for `field_radius`.

        The output is a tuple consisting the representation of the chosen receptive field, the PTM label, the amino
        acid sequence (as a character string) and optionally the RSA and SS3 labels and/or the kinase group.
        """
        pept_id, prot_pos, pept_pos, y = self.psite_pos[idx]
        prot_id = pept_id.split('_')[0] if '_' in pept_id else pept_id
        prot_len = len(self._protseqs[prot_id])
        prot_pos -= 1  # Detract 1 from the position, as python starts from 0 and the proteins start counting from 1
        pept_pos -= 1

        # Make sure representations are of the same shape by padding when necessary
        # This only occurs when the requested receptive field length exceeds the N- or C-terminus of the protein

        rf_len = 2 * self.radius + 1
        nterm, cterm = prot_pos - self.radius, prot_pos + self.radius

        # First make unpadded representations/sequences
        rf_seq = self._protseqs[prot_id][max(0, nterm):cterm + 1]
        rf_rep = self.data[pept_id][max(0, pept_pos - self.radius):pept_pos + self.radius + 1]

        padding = torch.zeros((rf_len, rf_rep.shape[1]))  # Make a zero tensor with the correct dimensions

        # Pad the tensors and adjust the string representation of the protein sequence
        if len(rf_seq) < rf_len:
            nseq_adjust, cseq_adjust = "", ""  # Will be added to the string representation of the sequence
            if nterm < 0:  # If padding is needed at the N-terminus
                pad = abs(nterm)
                nseq_adjust += "." * pad
                if cterm >= prot_len:  # In case of very short peptides, both termini may need padding
                    padding[pad:pad + len(rf_seq), :] = rf_rep
                    pad = cterm - prot_len + 1
                    cseq_adjust += "." * pad
                else:
                    padding[pad:, :] = rf_rep

            else:  # If padding is needed at the C-terminus only
                pad = cterm - prot_len + 1
                cseq_adjust += "." * pad
                padding[:len(rf_seq), :] = rf_rep

            rf_seq = nseq_adjust + rf_seq + cseq_adjust

        else:
            padding = rf_rep.float()  # If no padding is needed, tensor.dtype may still need converting to float

        # Lastly, for the CNN model, the tensor must be of the shape [channel, length], instead of [length, channel]
        X = padding.permute(1, 0)

        output = (X, y, rf_seq)

        # DSSP labels are added, if specified
        if self.dssp_dict is not None:
            rsa = find_sec_str(self.dssp_dict, prot_id, prot_pos, 'rsa')
            output += (rsa,)
            ss3 = find_sec_str(self.dssp_dict, prot_id, prot_pos, 'ss3')
            output += (ss3,)

        # Kinase family is added, if specified
        if self.kinase_spec is not None:
            kinase = find_kinase(self.kinase_spec, prot_id, prot_pos + 1, y)
            output += (kinase,)

        return output


class ReInitPSiteDataset(Dataset):
    """
    In case modifications need to be applied to the PSiteDataset instance, it is easier to first re-initialize the
    dataset for easier access to the data samples.
    """
    def __init__(self, rf_embeddings: list, labels: list, rf_sequences: list,
                 rsa: list = None, ss3: list = None, kinase: list = None):
        self.data = rf_embeddings
        self.labels = labels
        self.seqs = rf_sequences
        self.rsa = rsa
        self.ss3 = ss3
        self.kinase = kinase

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        output = (self.data[item], self.labels[item], self.seqs[item])
        if self.rsa is not None:
            output += (self.rsa[item], self.ss3[item])

        if self.kinase is not None:
            output += (self.kinase[item],)

        return output

    def __add__(self, other):
        """ Combine datasets or subsets """

        # Check whether both datasets can be combined
        if not (isinstance(other, ReInitPSiteDataset) or isinstance(other, PSiteSubset)):
            raise ValueError("Can only concatenate ReInitPSiteDataset instances or its subclasses")

        assert type(self.rsa) == type(other.rsa) and type(self.kinase) == type(other.kinase), \
            "The datasets do not contain the same types of secondary labels"

        data = self.data + other.data
        labels = self.labels + other.labels
        seqs = self.seqs + other.seqs
        rsa = None if self.rsa is None else self.rsa + other.rsa
        ss3 = None if self.ss3 is None else self.ss3 + other.ss3
        kinase = None if self.kinase is None else self.kinase + other.kinase

        return ReInitPSiteDataset(data, labels, seqs, rsa, ss3, kinase)

    def create_subwindow(self, new_radius: int):
        """
        Re-initialize the dataset but with smaller receptive fields. Prevents recreating the entire dataset from scratch
        during e.g. hyperparameter analysis. Input is the new, smaller radius. Output is a ReInitPSiteDataset object.
        """
        old_radius = len(self.seqs[0]) // 2
        assert new_radius <= old_radius, "New radius is not smaller than or equal to old radius"

        start_index = old_radius - new_radius
        stop_index = start_index + 2 * new_radius + 1

        for i in range(len(self)):
            self.data[i] = self.data[i][:, start_index:stop_index]
            self.seqs[i] = self.seqs[i][start_index:stop_index]


class PSiteSubset(ReInitPSiteDataset):
    """
    Create a subset of the dataset based on a list of select indices
    """
    def __init__(self, dataset, indices):
        super().__init__([dataset.data[i] for i in indices],
                         [dataset.labels[i] for i in indices],
                         [dataset.seqs[i] for i in indices])
        self.rsa = None if dataset.rsa is None else [dataset.rsa[i] for i in indices]
        self.ss3 = None if dataset.ss3 is None else [dataset.ss3[i] for i in indices]
        self.kinase = None if dataset.kinase is None else [dataset.kinase[i] for i in indices]


def balance_dataset(dataset: ReInitPSiteDataset):
    """
    Due to the high data imbalance (much more negative samples than positive), the dataset is balanced by randomly
    subsampling an equal number of data points from the negative dataset or re-sampling from the positive dataset.
    """
    random.seed(1)

    pos_idx, neg_idx = [], []

    for idx in range(len(dataset.labels)):
        pos_idx.append(idx) if dataset.__getitem__(idx)[1] == 1 else neg_idx.append(idx)

    sample_size = len(pos_idx)
    print(f"Before:\nPositive sample size: {sample_size}\nNegative sample size: {len(neg_idx)}")

    # If the positive dataset is large enough, subsample from the negative dataset, else re-sample from the positive one
    n = sample_size if sample_size >= 3000 else max(len(neg_idx), len(pos_idx))
    if n != sample_size:
        rem = abs(n - sample_size)
        pos_idx.extend(random.choices(pos_idx, k=rem))

    random.shuffle(neg_idx)
    neg_idx = neg_idx[:n]

    sampled_idx = pos_idx + neg_idx
    random.shuffle(sampled_idx)

    new_dataset = PSiteSubset(dataset, sampled_idx)

    return new_dataset


def get_subset_by_motif(dataset: ReInitPSiteDataset):
    """
    In case you want to use a subset of the data based on a specific recognition motif in the Serine/Threonine
    phosphoproteome. Currently only group samples with the P+1 proline motif and the remaining samples.
    """
    cl1, cl2 = [], []

    radius = len(dataset.seqs[0]) // 2

    for i in range(len(dataset.labels)):
        seq = dataset.__getitem__(i)[2]
        assert seq[radius] in 'ST', "Clustering by motif only possible for Ser/Thr phosphorylation data"

        # Cluster 1: +1 Pro
        if seq[radius + 1] == 'P':
            cl1.append(i)
        # Cluster 2: All others
        else:
            cl2.append(i)

    motif1 = PSiteSubset(dataset, cl1)
    motif2 = PSiteSubset(dataset, cl2)

    return motif1, motif2


def get_subset_by_rsa(dataset: ReInitPSiteDataset):
    """
    In case you want to use a subset of the data based on RSA score. Returns three subsets: Accessible (RSA >= 0.8),
    Interface (0.2 <= RSA < 0.8) and Buried (0.2 > RSA).
    """
    assert dataset.rsa is not None, "RSA labels not added to the dataset initialization"

    data_a, data_i, data_b = [], [], []

    for i in range(len(dataset.labels)):
        rsa = dataset.__getitem__(i)[3]
        if rsa != 'n/a':
            rsa = float(rsa)

            # Subset Accessible
            if rsa >= 0.8:
                data_a.append(i)
            # Subset Interface
            elif 0.2 <= rsa < 0.8:
                data_i.append(i)
            # Subset Buried
            else:
                data_b.append(i)

    subset_a = PSiteSubset(dataset, data_a)
    subset_i = PSiteSubset(dataset, data_i)
    subset_b = PSiteSubset(dataset, data_b)

    return subset_a, subset_i, subset_b


def get_subset_by_kinase(dataset: ReInitPSiteDataset):
    """
    Split the data into subsets based on the specific protein kinase group that binds and phosphorylates the P-site.
    Only applicable to Ser/Thr phosphorylation data. Positions with no kinase assigned are omitted. Kinase families:
    'AGC', 'Alpha', 'CAMK', 'CK1', 'CMGC', 'FAM20C', 'PDHK', 'PIKK', 'STE', 'TKL' and 'Other'.
    """
    assert dataset.kinase is not None, "Cannot subset by kinase without kinase data"

    radius = len(dataset.seqs[0]) // 2
    dssp_dict = False if dataset.rsa is None else True
    kinase_dict = {}
    
    for i in range(len(dataset.labels)):
        if dssp_dict:
            _, _, seq, *_, kinase = dataset.__getitem__(i)
        else:
            _, _, seq, kinase = dataset.__getitem__(i)
        
        assert seq[radius] in 'ST', "Clustering by Kinase family only allowed on Ser/Thr phosphorylation data"

        if kinase not in ('Undefined', 'N/A'):
            kinase_dict.setdefault(kinase, []).append(i)

    return tuple([PSiteSubset(dataset, indices) for indices in kinase_dict.values()])


def re_init_dataset(dataset: PSiteDataset):
    """
    Make a new instance of the same dataset that is simplified and that contains only the necessary data needed for
    further manipulation: i.e. embedding and sequence lengths are restricted to the receptive field instead of keeping
    the full peptide information.
    """
    data = [dataset[i] for i in range(len(dataset))]

    # None of the optional labels were stated in the original Dataset
    if dataset.dssp_dict is None and dataset.kinase_spec is None:
        data_points, labels, sequences = map(list, zip(*data))
        new_dataset = ReInitPSiteDataset(data_points, labels, sequences)

    else:
        # Only the structural labels were specified
        if dataset.kinase_spec is None:
            data_points, labels, sequences, rsa, ss3 = map(list, zip(*data))
            new_dataset = ReInitPSiteDataset(data_points, labels, sequences, rsa, ss3)
        # Only the kinase labels were specified
        elif dataset.dssp_dict is None:
            data_points, labels, sequences, kinase = map(list, zip(*data))
            new_dataset = ReInitPSiteDataset(data_points, labels, sequences, kinase=kinase)
        # All optional labels were specified
        else:
            data_points, labels, sequences, rsa, ss3, kinase = map(list, zip(*data))
            new_dataset = ReInitPSiteDataset(data_points, labels, sequences, rsa, ss3, kinase)

    return new_dataset


def restrict_radius(dataset: PSiteDataset, radius: int):
    """
    Makes a new instance of an already existing dataset with the new, smaller radius. Function is mainly used
    to compare different receptive field lengts for hyperparameter testing. Does not assume any additional labels
    such as structural info or kinase group, but limits the dataset to the representations, PTM label and AA sequence.

    Returns the new dataset as a `ReinitPSiteDataset` instance where __getitem__ yields: (representation, label, seq)
    """
    print("Constructing radius-restricted Dataset...")
    psite = len(dataset[0][2]) // 2
    assert radius <= psite, "The given radius is bigger than the original radius"

    data, labels, seqs = [], [], []

    for i in range(len(dataset)):
        x = dataset[i][0].permute(1, 0)
        x = x[psite - radius:psite + radius + 1]
        data.append(x.permute(1, 0))
        labels.append(dataset[i][1])
        seqs.append(dataset[i][2][psite - radius: psite + radius + 1])

    return ReInitPSiteDataset(data, labels, seqs)
