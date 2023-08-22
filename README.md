# Master's Thesis 2022-2023
## Summary
This repository contains the scripts necessary to create the phosphorylation site datasets needed for P-site prediction, run the *PSitePredict* deep learning algorithm and perform comparative analyses/judge performances.

- *scripts*: contains all scripts for P-site dataset creation, model training and analyses.
- *PSite_Files*: contains the raw FASTA files for the serine/threonine and tyrosine datasets, split into separate files for training (85%), testing (10%) and validation (5%).
- *Kinase_Specificity*: contains three .json files with Python dictionaries linking P-sites (i.e. substrates) to their Kinase Group. Version 2 is the result of running **Kinase_Specificity.py**, Version 1 is deprecated and **TopRank** is the result of simply assigning the Kinase Group with rank number 1 to the susbtrate.
