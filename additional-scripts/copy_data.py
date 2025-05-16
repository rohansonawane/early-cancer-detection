import os
import shutil
import pandas as pd

def copy_data():
    """Copy necessary data from the original project"""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(current_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Copy KEGG pathways file
    kegg_dir = os.path.join(current_dir, 'KEGG_pathways')
    os.makedirs(kegg_dir, exist_ok=True)
    shutil.copy(
        os.path.join(current_dir, '..', 'KEGG_pathways', '20230205_kegg_hsa.gmt'),
        os.path.join(kegg_dir, '20230205_kegg_hsa.gmt')
    )
    
    # Copy data for each cancer type
    cancer_types = ['BLCA', 'LIHC', 'PRAD', 'BRCA', 'AML', 'WT']
    
    for cancer_type in cancer_types:
        # Create cancer-specific directory
        cancer_dir = os.path.join(data_dir, f'{cancer_type}_data')
        os.makedirs(cancer_dir, exist_ok=True)
        
        # Copy data files
        data_files = [
            'mRNA_data.csv',
            'miRNA_data.csv',
            'snv_data.csv',
            'response.csv'
        ]
        
        for file in data_files:
            src = os.path.join(current_dir, '..', f'{cancer_type}_data', file)
            dst = os.path.join(cancer_dir, file)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f'Copied {src} to {dst}')
            else:
                print(f'Warning: {src} does not exist')

if __name__ == '__main__':
    copy_data() 