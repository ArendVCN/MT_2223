import sys
from AddStructureData import add_dssp2datadir
from AccessCorrectFolders import load_in_dicts
from Bio.Data.IUPACData import protein_letters

"""
Run from the command line.

Takes as a single argument the type of embedding (onehot, esm, carp, alphafold-t, alphafold-nt) to modify with
structural data and will create a new directory structured exactly like the original directory, but containing the
structurally modified versions of the original representations.
"""
dssp_dict, _ = load_in_dicts(protein_letters + "U", "all_amino_acids_numerical.tsv")

add_dssp2datadir(sys.argv[1], dssp_dict)
