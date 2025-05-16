import os
import pandas as pd
import numpy as np

def extract_center_id(tcga_id):
    """Extract the center ID from TCGA sample ID."""
    parts = tcga_id.split('-')
    if len(parts) >= 2:
        return parts[1]  # The second part of TCGA ID typically represents the tissue source site
    return 'unknown'

def get_sample_id_column(df):
    """Find the column containing sample IDs."""
    possible_names = ['Case_ID', 'case_id', 'Sample_ID', 'sample_id', 'ID', 'id']
    for name in possible_names:
        if name in df.columns:
            return name
    # If no standard name found, assume the index contains sample IDs
    return df.index.name if df.index.name else df.index

def combine_small_batches(batch_info, min_samples_per_batch=2):
    """Combine batches with fewer than min_samples_per_batch samples."""
    # Count samples per batch
    batch_counts = batch_info['batch'].value_counts()
    
    # Identify small batches
    small_batches = batch_counts[batch_counts < min_samples_per_batch].index
    
    if len(small_batches) > 0:
        print(f"Combining {len(small_batches)} small batches...")
        
        # Create a mapping for small batches
        batch_mapping = {}
        for batch in small_batches:
            # Find the nearest larger batch
            larger_batches = batch_counts[batch_counts >= min_samples_per_batch]
            if len(larger_batches) > 0:
                nearest_batch = larger_batches.index[0]
                batch_mapping[batch] = nearest_batch
            else:
                # If no larger batches, combine all small batches into one
                batch_mapping[batch] = max(batch_counts.index) + 1
        
        # Apply the mapping
        batch_info['batch'] = batch_info['batch'].map(lambda x: batch_mapping.get(x, x))
        
        # Reassign batch numbers to be consecutive
        unique_batches = sorted(batch_info['batch'].unique())
        batch_mapping = {old: new for new, old in enumerate(unique_batches, 1)}
        batch_info['batch'] = batch_info['batch'].map(batch_mapping)
    
    return batch_info

def generate_batch_info(data_dir, cancer_type):
    """Generate batch information for a specific cancer type."""
    print(f"Generating batch info for {cancer_type}...")
    
    # Read response file to get sample IDs
    response_file = os.path.join(data_dir, f'{cancer_type}_data', 'response.csv')
    if not os.path.exists(response_file):
        print(f"Response file not found for {cancer_type}")
        return
    
    # Read sample IDs
    response_df = pd.read_csv(response_file)
    id_column = get_sample_id_column(response_df)
    
    if isinstance(id_column, pd.Index):
        sample_ids = id_column.values
    else:
        sample_ids = response_df[id_column].values
    
    # Extract center IDs and create batch mapping
    center_ids = [extract_center_id(sample_id) for sample_id in sample_ids]
    unique_centers = sorted(set(center_ids))
    center_to_batch = {center: idx + 1 for idx, center in enumerate(unique_centers)}
    
    # Create batch labels
    batch_labels = [center_to_batch[center] for center in center_ids]
    
    # Create batch info DataFrame
    batch_info = pd.DataFrame({
        'sample_id': sample_ids,
        'center_id': center_ids,
        'batch': batch_labels
    })
    
    # Combine small batches
    batch_info = combine_small_batches(batch_info, min_samples_per_batch=2)
    
    # Save batch info
    output_file = os.path.join(data_dir, f'{cancer_type}_data', 'batch_info.csv')
    batch_info.to_csv(output_file, index=False)
    print(f"Created batch info file: {output_file}")
    print(f"Number of batches: {len(batch_info['batch'].unique())}")
    print(f"Samples per batch:")
    print(batch_info['batch'].value_counts().sort_index())
    print()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    cancer_types = ['BLCA', 'LIHC', 'PRAD', 'BRCA', 'AML', 'WT']
    
    for cancer_type in cancer_types:
        generate_batch_info(data_dir, cancer_type)

if __name__ == '__main__':
    main() 