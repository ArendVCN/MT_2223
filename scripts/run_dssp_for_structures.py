from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import os

'''
Create a tab-delimited text file with the structural (DSSP) information for each amino acid position for each protein
sequence within the datasets.
Only gives information of the data is known, otherwise leaves the data out.

Returns a print out of: 
- the protein ID,
- amino acid position, 
- residue name, 
- solvant accessibility (as A, I, or B),
- eight-state secondary structure denomination,
- three-state secondary structure denomination and
- residue solvent accessibility score (as a percentage)
'''

pdb_root = '/data/home/jasperz/data/ALPHAFOLD_DB/'
input_fastas = ['/home/arendvc/PSite_Files/Ram22_pep_filtered_ST_train.fasta',
                '/home/arendvc/PSite_Files/Ram22_pep_filtered_ST_test.fasta',
                '/home/arendvc/PSite_Files/Ram22_pep_filtered_ST_valid.fasta']
out = '/home/arendvc/dssp/all_amino_acids_numerical.tsv'

all_files = {}
for file in os.listdir(pdb_root + 'HUMAN_UNZIPPED/'):
    if file.endswith('.pdb'):
        all_files.setdefault(file.split('-')[1], []).append(pdb_root + 'HUMAN_UNZIPPED/' + file)
for file in os.listdir(pdb_root + 'manual/'):
    if file.endswith('.pdb'):
        all_files.setdefault(file.split('_')[0], []).append(pdb_root + 'manual/' + file)

fasta_dict = {}
for input_fasta in input_fastas:
    fasta_dict.update({rec.id.split('|')[1] if '|' in rec.id else rec.id: [x for x in str(rec.seq) if x.isalpha()] for rec in SeqIO.parse(open(input_fasta), "fasta")})

map_ss = {
    '-': 'C',
    'G': 'H',
    'H': 'H',
    'I': 'H',
    'T': 'C',
    'E': 'E',
    'B': 'C',
    'S': 'C'
}

write_to = open(out, 'w')
print('ACC_ID\tUP_POS\tresidue\tAA\tSsec8\tSsec3', file=write_to)

for prot_id, seq in fasta_dict.items():
    if prot_id in all_files:
        f = all_files[prot_id][0]
        p = PDBParser()
        structure = p.get_structure(prot_id, f)
        model = structure[0]
        dssp = DSSP(model, f)

        for pos in range(len(seq)):
            try:
                key = list(dssp.keys())[pos+1]
                sec_str_dssp = dssp[key][2]
                rel_solv_acc = dssp[key][3]
                sec_str_3 = map_ss[sec_str_dssp]
                sec_str_8 = sec_str_dssp
                sol_acc = 'B' if rel_solv_acc <= 0.2 else 'A' if rel_solv_acc >= 0.8 else 'I'
                print(f'{prot_id}\t{pos}\t{seq[pos][0]}\t{sol_acc}\t{sec_str_8}\t{sec_str_3}\t{rel_solv_acc:.2f}',
                      file=write_to)
            except IndexError:
                print(prot_id)
                continue
    else:
        print(prot_id+' not found.')
