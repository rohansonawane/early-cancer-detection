import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from inmoose.pycombat import pycombat_norm
import networkx as nx
import logging

class MultiOmicsProcessor:
    def __init__(self, kegg_pathways_file):
        """
        Initialize the MultiOmicsProcessor
        
        Args:
            kegg_pathways_file (str): Path to KEGG pathways annotation file
        """
        self.kegg_pathways = self._load_kegg_pathways(kegg_pathways_file)
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=10)
        self.logger = logging.getLogger(__name__)
        
    def _load_kegg_pathways(self, file_path):
        """Load KEGG pathways from file"""
        pathways = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                pathway_id = parts[0].split('_')[0]
                genes = parts[2:]
                pathways[pathway_id] = genes
        return pathways
    
    def impute_missing_values(self, data):
        """Impute missing values using KNN"""
        self.logger.info(f"Imputing missing values for {len(data)} samples")
        imputed_data = pd.DataFrame(
            self.imputer.fit_transform(data),
            index=data.index,
            columns=data.columns
        )
        self.logger.info(f"After imputation: {len(imputed_data)} samples")
        return imputed_data
    
    def correct_batch_effects(self, data, batch_labels):
        """Correct batch effects using ComBat"""
        self.logger.info(f"Correcting batch effects for {len(data)} samples")
        # Transpose data for combat (samples in columns)
        data_t = data.T
        # Apply combat correction
        corrected_data = pycombat_norm(
            data=data_t.values,
            batch=batch_labels,
            parametric=True
        )
        # Transpose back and create DataFrame
        result = pd.DataFrame(
            corrected_data.T,
            index=data.index,
            columns=data.columns
        )
        self.logger.info(f"After batch correction: {len(result)} samples")
        return result
    
    def process_mrna_data(self, mrna_data, batch_labels=None):
        """Process mRNA expression data"""
        self.logger.info(f"Processing mRNA data for {len(mrna_data)} samples")
        
        # Impute missing values
        mrna_imputed = self.impute_missing_values(mrna_data)
        
        # Correct batch effects if batch labels are provided
        if batch_labels is not None:
            mrna_corrected = self.correct_batch_effects(mrna_imputed, batch_labels)
        else:
            mrna_corrected = mrna_imputed
            
        # Standardize the data
        mrna_scaled = self.scaler.fit_transform(mrna_corrected)
        result = pd.DataFrame(mrna_scaled, index=mrna_data.index, columns=mrna_data.columns)
        self.logger.info(f"After mRNA processing: {len(result)} samples")
        return result
    
    def process_mirna_data(self, mirna_data, batch_labels=None):
        """Process miRNA expression data"""
        self.logger.info(f"Processing miRNA data for {len(mirna_data)} samples")
        
        # Impute missing values
        mirna_imputed = self.impute_missing_values(mirna_data)
        
        # Correct batch effects if batch labels are provided
        if batch_labels is not None:
            mirna_corrected = self.correct_batch_effects(mirna_imputed, batch_labels)
        else:
            mirna_corrected = mirna_imputed
            
        # Standardize the data
        mirna_scaled = self.scaler.fit_transform(mirna_corrected)
        result = pd.DataFrame(mirna_scaled, index=mirna_data.index, columns=mirna_data.columns)
        self.logger.info(f"After miRNA processing: {len(result)} samples")
        return result
    
    def process_snv_data(self, snv_data):
        """Process SNV data"""
        self.logger.info(f"Processing SNV data for {len(snv_data)} samples")
        # Convert to binary features
        snv_binary = (snv_data > 0).astype(int)
        self.logger.info(f"After SNV processing: {len(snv_binary)} samples")
        return snv_binary
    
    def create_pathway_features(self, mrna_data, mirna_data):
        """Create pathway-based features"""
        self.logger.info(f"Creating pathway features for {len(mrna_data)} samples")
        pathway_features = {}
        
        for pathway_id, genes in self.kegg_pathways.items():
            # Get genes present in both pathway and data
            mrna_genes = set(mrna_data.columns).intersection(set(genes))
            mirna_genes = set(mirna_data.columns).intersection(set(genes))
            
            if mrna_genes:
                pathway_features[f'{pathway_id}_mrna'] = mrna_data[list(mrna_genes)].mean(axis=1)
            if mirna_genes:
                pathway_features[f'{pathway_id}_mirna'] = mirna_data[list(mirna_genes)].mean(axis=1)
        
        result = pd.DataFrame(pathway_features)
        self.logger.info(f"After pathway feature creation: {len(result)} samples")
        return result
    
    def integrate_data(self, mrna_data, mirna_data, snv_data, batch_labels=None):
        """Integrate all omics data"""
        self.logger.info(f"Integrating data for {len(mrna_data)} samples")
        
        # Process individual data types
        mrna_processed = self.process_mrna_data(mrna_data, batch_labels)
        mirna_processed = self.process_mirna_data(mirna_data, batch_labels)
        snv_processed = self.process_snv_data(snv_data)
        
        # Create pathway features
        pathway_features = self.create_pathway_features(mrna_processed, mirna_processed)
        
        # Combine all features
        integrated_data = pd.concat([
            mrna_processed,
            mirna_processed,
            snv_processed,
            pathway_features
        ], axis=1)
        
        self.logger.info(f"Final integrated data: {len(integrated_data)} samples")
        return integrated_data 