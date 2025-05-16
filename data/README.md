# Data Directory Structure

This directory contains the data files for the early cancer detection project.

## Directory Structure

- `BLCA_data/`: Contains data files for Bladder Urothelial Carcinoma (BLCA)
  - `mRNA_data.csv`: mRNA expression data
  - `miRNA_data.csv`: miRNA expression data
  - `snv_data.csv`: Single Nucleotide Variation data
  - `response.csv`: Response labels
  - `batch_info.csv`: Batch information

## Data Format

Each cancer type directory contains the following files:
- mRNA_data.csv: Gene expression data
- miRNA_data.csv: miRNA expression data
- snv_data.csv: Mutation data
- response.csv: Binary labels (0: Normal, 1: Cancer)
- batch_info.csv: Batch information for normalization

## Note

The full dataset is not included in this repository due to size limitations.
Please contact the authors for access to the complete dataset. 