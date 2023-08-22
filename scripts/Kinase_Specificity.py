import json
import pandas as pd
from statistics import median

'''
Assign Kinase Groups to P-sites depending on which kinases catalyse the site the most with the highest specificity.
The comma-separated file `KinaseSubstrateTable.csv` from which the information has been extracted was provided by 
Johnson, J.L., Yaron, T.M., Huntsman, E.M. et al. (2023).
'''

# Open the file with data from the Kinome study
with open('/home/arendvc/Kinase_Specificity/KinaseSubstrateTable.csv', 'r') as infile:
    csv_file = pd.read_csv(infile, delimiter=';')

csv_file['Psite_Pos'] = csv_file['Phosphosite'].str.extract(r'(\d+)').astype(int)  # Extract the positions as integers
csv_file = csv_file[csv_file['Database'].isin(['Ochoa', 'Ochoa,PSP-LT'])]  # Prevent duplicates by using 1 database

# Constrict dataframe to the relevant data
df1 = pd.DataFrame(csv_file).loc[:, ['Uniprot Primary Accession', 'Psite_Pos', 'SITE_+/-7_AA']]
df2 = pd.DataFrame(csv_file).iloc[:, 11:-11]
df = pd.concat([df1, df2], axis=1)
df = df.drop_duplicates(subset=['Uniprot Primary Accession', 'Psite_Pos'])  # Remove remaining duplicates

# Dictionary with Protein kinase groups as keys and a list of kinases as value
kinase_groups = {'AGC': ['AKT', 'GRK', 'LATS', 'MASTL', 'MRCK',
                         'MSK', 'NDR', 'P70', 'P90', 'PDK',
                         'PKA', 'PKC', 'PKG', 'PKN', 'PRKX',
                         'ROCK', 'RSK', 'SGK', 'YANK'],
                 'ALPHA': ['ALPHA', 'CHA', 'EEF'],
                 'CAMK': ['AMPKA', 'BRSK', 'CAMK1', 'CAMK2', 'CAMK4',
                          'CAML', 'CHK', 'CRIK', 'DAPK', 'DCAM',
                          'DMPK', 'DRAK', 'HUNK', 'LKB', 'MAPK',
                          'MARK', 'MELK', 'MNK', 'MYLK', 'NIM',
                          'NUAK', 'PAS', 'PHK', 'PIM', 'PRKD',
                          'QIK', 'QSK', 'SIK', 'SKML', 'SMML',
                          'SNRK', 'SSTK', 'STK', 'TSSK'],
                 'CK1': ['CK1', 'TTBK', 'VRK'],
                 'CMGC': ['CDK', 'CLK', 'DYRK', 'ERK', 'GSK',
                          'HIPK', 'ICK', 'JNK', 'MAK', 'MOK',
                          'NLK', 'P38', 'PRP4', 'SRP', ],
                 'FAM20C': ['FAM'],
                 'Other': ['AAK', 'AUR', 'BIKE', 'BUB', 'CAMKK',
                           'CDC', 'CK2', 'DSTYK', 'GAK', 'GCN',
                           'HASP', 'HRI', 'IKK', 'IRE', 'KIS',
                           'MOS', 'MPS', 'NEK', 'PBK', 'PERK',
                           'PIN', 'PKR', 'PLK', 'PRPK', 'SBK',
                           'TBK', 'TLK', 'TTK', 'ULK', 'WNK'],
                 'PDHK': ['BCK', 'PDH'],
                 'PIKK': ['ATM', 'ATR', 'DNA', 'MTOR', 'SMG'],
                 'STE': ['ASK', 'COT', 'GCK', 'HGK', 'HPK',
                         'KHS', 'LOK', 'MAP3', 'MEK', 'MINK',
                         'MST', 'MYO', 'NIK', 'OSR', 'PAK',
                         'SLK', 'STL', 'TAO', 'TNI', 'YSK'],
                 'TKL': ['ACV', 'ALK', 'ANK', 'BMP', 'BRAF',
                         'DLK', 'IRA', 'LRR', 'MLK', 'RAF',
                         'RIP', 'TAK', 'TGF', 'ZAK']
                 }

reverse_kinase_groups = {k: group for group, kinase_list in kinase_groups.items() for k in kinase_list}


# Extract for each kinase the percentile and rank columns
percentile_cols = []
rank_cols = []

for column in df2.iloc[:, 2:].columns:
    if column.endswith('rank'):
        rank_cols.append(column)
    else:
        percentile_cols.append(column)


# Construct a dictionary with for each protein substrate and P-site position the relevant kinase group
kinase_psite_dict = {}

for index, row in df.iterrows():
    row_dict = row.to_dict()

    kf_output = 'Non-specific'
    if row_dict['promiscuity_index'] > 100:
        pass  # Will skip very promiscuous substrates and immediately classify them as 'Non-specific'
    else:
        kinases = []
        kinase_ranks = []
        kinase_group_counts = {}
        kinase_rank_medians = {}

        for i in range(len(percentile_cols)):
            if row_dict['promiscuity_index'] <= 30:  # For substrates with high kinase specificity (<10% of kinases)
                if row_dict[percentile_cols[i]] >= 90 and row_dict[rank_cols[i]] <= row_dict['promiscuity_index']:
                    kinase = percentile_cols[i].split('_')[0]
                    kinases.append(kinase)
                    kinase_ranks.append(row_dict[rank_cols[i]])
            elif row_dict['promiscuity_index'] <= 60:  # For substrates with moderate specificity (<20% of kinases)
                if row_dict[percentile_cols[i]] > 95 and row_dict[rank_cols[i]] <= 20:
                    kinase = percentile_cols[i].split('_')[0]
                    kinases.append(kinase)
                    kinase_ranks.append(row_dict[rank_cols[i]])
            else:  # For substrates with low specificity (<33% of kinases)
                if row_dict[percentile_cols[i]] > 98 and row_dict[rank_cols[i]] <= 30:
                    kinase = percentile_cols[i].split('_')[0]
                    kinases.append(kinase)
                    kinase_ranks.append(row_dict[rank_cols[i]])

        # If the previous methods did not select any kinase, try the top 10-ranked kinases for the substrate
        # Likely these substrates had no kinases with percentile scores above 90 (promiscuity = 0)
        if not kinases:
            for i in range(len(percentile_cols)):
                if row_dict[rank_cols[i]] <= 10 and row_dict[percentile_cols[i]] >= 85:
                    kinases.append(rank_cols[i].split('_')[0])
                    kinase_ranks.append(row_dict[rank_cols[i]])

        if kinases:
            # Count which kinase protein group the most members binding the substrate P-site
            # Also keeps the ranks in case >1 groups have an equal number of binding members
            for i in range(len(kinases)):
                for kin_group in reverse_kinase_groups:
                    if kinases[i].startswith(kin_group):
                        key = reverse_kinase_groups[kin_group]
                        kinase_group_counts[key] = kinase_group_counts.setdefault(key, 0) + 1
                        kinase_rank_medians.setdefault(key, []).append(kinase_ranks[i])

            # The kinase group with the maximum number of binders AND the lowest median rank is chosen
            max_kinases = max(kinase_group_counts.values())
            best_kinase_rank = float('inf')

            for key, value in kinase_group_counts.items():
                if value == max_kinases:
                    rank_median = median(kinase_rank_medians[key])
                    if rank_median < best_kinase_rank:
                        best_kinase_group = rank_median
                        kf_output = key

    prot_id = row_dict['Uniprot Primary Accession']
    kinase_psite_tuple = (row_dict['Psite_Pos'], kf_output)
    kinase_psite_dict.setdefault(prot_id, []).append(kinase_psite_tuple)


# Save the created dictionary to a file for later use
with open('/home/arendvc/Kinase_Specificity/KinasePsiteDictTopRank.json', 'w') as outfile:
    json.dump(kinase_psite_dict, outfile)
