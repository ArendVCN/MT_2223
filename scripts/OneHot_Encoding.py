import os
import sys
import torch
import torch.nn.functional as F
from Bio import SeqIO
from Bio.Data.IUPACData import protein_letters

'''
Create and store one-hot encoded representations of the protein sequences within a FASTA file as separate .pt files 
within a designated directory.
'''


class OneHot:
    def __init__(self, unlab_fasta_file, alphabet):
        """
        An instance of OneHot needs a normal fasta file of proteins to encode as input + the protein alphabet that is
        used to encode it.
        """
        self.labels = {char: lab for lab, char in enumerate(alphabet)}
        self.data = self.__store_reps_in_dict(unlab_fasta_file)
        self.filename = str(unlab_fasta_file).split("/")[-1].split(".")[0]

    def _seq2onehot(self, seq):
        """
        Transforms a simple protein sequence into a One-Hot encoded LongTensor/torch.int64 item
        by way of first encoding amino acids as integers numbered 0-20 and then transforming this tensor
        into a one-hot representation through the torch.nn.functional.one_hot function.

        Returns the one-hot encoded tensor corresponding to the input sequence.
        """
        output = torch.tensor([self.labels[aa.upper()] for aa in seq])
        return F.one_hot(output, num_classes=len(self.labels))

    def __store_reps_in_dict(self, fasta_file):
        """
        Stores the One Hot representation for each sequence from an input fasta file into a dictionary
        with the Protein ID as key.

        Returns the mentioned dictionary.
        """
        reps = {}

        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            seq_id, rep = seq_record.id, self._seq2onehot(seq_record.seq)
            reps[seq_id] = rep

        return reps

    def store_data(self, dir_path):
        """
        This method saves the dictionary-stored one-hot representations as .pt files inside a new directory located in
        the given parent directory `dir_path`.
        """
        dir_name = self.filename.replace("pep_filtered_ST_", "")
        path = os.path.join(dir_path, dir_name)
        # print(path)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            print("File or directory with this name already exists at this location.")

        for pid, rep in self.data.items():
            fh = path + "/" + pid + ".pt"
            # print(fh)
            torch.save(rep, fh)


# Instantiate an instance of the OneHot class and save it to a given directory
test_onehot = OneHot(sys.argv[1], protein_letters + "U")  # There are a select few protein with SelenoCysteine (U)
test_onehot.store_data(sys.argv[2])
