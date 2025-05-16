import unittest
import numpy as np
import pandas as pd
from utils.component_analysis import ComponentAnalysis, MOFA
from utils.biomarker_ranking import BiomarkerRanker
from utils.multi_omics_processor import MultiOmicsProcessor

class TestComponentAnalysis(unittest.TestCase):
    def setUp(self):
        self.n_components = 3
        self.ca = ComponentAnalysis(n_components=self.n_components)
        self.mofa = MOFA(n_components=self.n_components)
        
        # Create sample data
        np.random.seed(42)
        self.mrna_data = pd.DataFrame(np.random.randn(100, 50))
        self.mirna_data = pd.DataFrame(np.random.randn(100, 30))
        self.snv_data = pd.DataFrame(np.random.randn(100, 20))
        
    def test_pca_fit_transform(self):
        # Test PCA fit and transform
        transformed_data, explained_variance = self.ca.fit_transform(
            [self.mrna_data, self.mirna_data, self.snv_data]
        )
        
        self.assertEqual(transformed_data.shape[1], self.n_components)
        self.assertEqual(len(explained_variance), self.n_components)
        self.assertTrue(np.all(explained_variance >= 0))
        self.assertTrue(np.all(explained_variance <= 1))
        
    def test_mofa_fit_transform(self):
        # Test MOFA fit and transform
        transformed_data, factor_importance = self.mofa.fit_transform(
            [self.mrna_data, self.mirna_data, self.snv_data]
        )
        
        self.assertEqual(transformed_data.shape[1], self.n_components)
        self.assertEqual(len(factor_importance), self.n_components)
        self.assertTrue(np.all(factor_importance >= 0))

class TestBiomarkerRanker(unittest.TestCase):
    def setUp(self):
        # Create sample pathway data
        self.pathway_data = {
            'pathway1': ['gene1', 'gene2', 'gene3'],
            'pathway2': ['gene4', 'gene5', 'gene6']
        }
        self.ranker = BiomarkerRanker(self.pathway_data)
        
        # Create sample data
        np.random.seed(42)
        self.mrna_data = pd.DataFrame(np.random.randn(100, 50))
        self.mirna_data = pd.DataFrame(np.random.randn(100, 30))
        self.snv_data = pd.DataFrame(np.random.randn(100, 20))
        self.labels = pd.Series(np.random.randint(0, 2, 100))
        
    def test_rank_biomarkers(self):
        # Test biomarker ranking
        top_n = 10
        ranked_biomarkers = self.ranker.rank_biomarkers(
            self.mrna_data,
            self.mirna_data,
            self.snv_data,
            self.labels,
            top_n=top_n
        )
        
        self.assertIn('mRNA', ranked_biomarkers)
        self.assertIn('miRNA', ranked_biomarkers)
        self.assertIn('SNV', ranked_biomarkers)
        
        for omics_type in ['mRNA', 'miRNA', 'SNV']:
            self.assertEqual(len(ranked_biomarkers[omics_type]), top_n)
            
    def test_pathway_enrichment(self):
        # Test pathway enrichment
        biomarkers = {
            'mRNA': ['gene1', 'gene2', 'gene4'],
            'miRNA': ['mir1', 'mir2'],
            'SNV': ['snp1', 'snp2']
        }
        
        enrichment = self.ranker.get_pathway_enrichment(biomarkers)
        
        self.assertIn('pathway1', enrichment)
        self.assertIn('pathway2', enrichment)
        self.assertTrue(all(0 <= p <= 1 for p in enrichment.values()))

class TestMultiOmicsProcessor(unittest.TestCase):
    def setUp(self):
        # Create sample KEGG pathways file
        self.kegg_pathways = {
            'pathway1': ['gene1', 'gene2', 'gene3'],
            'pathway2': ['gene4', 'gene5', 'gene6']
        }
        self.processor = MultiOmicsProcessor(self.kegg_pathways)
        
        # Create sample data
        np.random.seed(42)
        self.mrna_data = pd.DataFrame(np.random.randn(100, 50))
        self.mirna_data = pd.DataFrame(np.random.randn(100, 30))
        self.snv_data = pd.DataFrame(np.random.randn(100, 20))
        
    def test_process_mrna_data(self):
        # Test mRNA data processing
        processed_data = self.processor.process_mrna_data(self.mrna_data)
        
        self.assertEqual(processed_data.shape, self.mrna_data.shape)
        self.assertTrue(np.all(processed_data >= 0))
        self.assertTrue(np.all(processed_data <= 1))
        
    def test_process_mirna_data(self):
        # Test miRNA data processing
        processed_data = self.processor.process_mirna_data(self.mirna_data)
        
        self.assertEqual(processed_data.shape, self.mirna_data.shape)
        self.assertTrue(np.all(processed_data >= 0))
        self.assertTrue(np.all(processed_data <= 1))
        
    def test_process_snv_data(self):
        # Test SNV data processing
        processed_data = self.processor.process_snv_data(self.snv_data)
        
        self.assertEqual(processed_data.shape, self.snv_data.shape)
        self.assertTrue(np.all(processed_data >= 0))
        self.assertTrue(np.all(processed_data <= 1))

if __name__ == '__main__':
    unittest.main() 