import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_processor import MultiOmicsProcessor
from models.enhanced_cancer_detector import EnhancedCancerDetector
from inmoose.pycombat import pycombat_norm
import json
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self, base_dir, cancer_type, test_size=0.1):
        """
        Initialize ModelTrainer
        
        Args:
            base_dir (str): Base directory containing the data
            cancer_type (str): Type of cancer to analyze
            test_size (float): Proportion of data to use for testing (default: 0.1)
        """
        self.base_dir = base_dir
        self.cancer_type = cancer_type
        self.test_size = test_size
        self.data_dir = os.path.join(base_dir, f'{cancer_type}_data')
        self.models_dir = os.path.join(base_dir, 'detect-cancer', 'models')
        self.results_dir = os.path.join(base_dir, 'detect-cancer', 'results', cancer_type)
        
        # Create necessary directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize data processor
        self.processor = MultiOmicsProcessor(os.path.join(base_dir, 'KEGG_pathways', '20230205_kegg_hsa.gmt'))
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        # Load data
        self.logger.info("Loading data files...")
        mrna_data = pd.read_csv(os.path.join(self.data_dir, 'mRNA_data.csv'), index_col=0)
        mirna_data = pd.read_csv(os.path.join(self.data_dir, 'miRNA_data.csv'), index_col=0)
        snv_data = pd.read_csv(os.path.join(self.data_dir, 'snv_data.csv'), index_col=0)
        labels = pd.read_csv(os.path.join(self.data_dir, 'response.csv'), index_col=0)
        
        self.logger.info(f"Initial sample counts: mRNA={len(mrna_data)}, miRNA={len(mirna_data)}, SNV={len(snv_data)}, labels={len(labels)}")
        
        # Ensure all data have the same samples
        common_samples = list(set(mrna_data.index) & set(mirna_data.index) & set(snv_data.index) & set(labels.index))
        self.logger.info(f"Number of common samples across all data types: {len(common_samples)}")
        
        # Filter data to keep only common samples
        mrna_data = mrna_data.loc[common_samples]
        mirna_data = mirna_data.loc[common_samples]
        snv_data = snv_data.loc[common_samples]
        labels = labels.loc[common_samples]
        
        # Process mRNA data
        mrna_processed = self.processor.process_mrna_data(mrna_data)
        
        # Process miRNA data
        mirna_processed = self.processor.process_mirna_data(mirna_data)
        
        # Process SNV data if available
        if not snv_data.empty:
            snv_processed = self.processor.process_snv_data(snv_data)
            # Combine features
            X = pd.concat([mrna_processed, mirna_processed, snv_processed], axis=1)
        else:
            # Combine features without SNV data
            X = pd.concat([mrna_processed, mirna_processed], axis=1)
            
        y = labels['response'].values
        
        self.logger.info(f"Final preprocessed data shape: X={X.shape}, y={len(y)}")
        self.logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Convert DataFrame to numpy array
        X = X.values
        
        return X, y
    
    def augment_data(self, X, y, noise_factor=0.05):
        """Apply data augmentation to increase training data"""
        X_aug = X.copy()
        y_aug = y.copy()
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_factor, X.shape)
        X_aug = X_aug + noise
        
        # Combine original and augmented data
        X_combined = np.vstack([X, X_aug])
        y_combined = np.concatenate([y, y_aug])
        
        self.logger.info(f"Data shape after augmentation: X={X_combined.shape}, y={len(y_combined)}")
        self.logger.info(f"Class distribution after augmentation: {np.bincount(y_combined)}")
        
        return X_combined, y_combined
    
    def split_data(self, X, y):
        """Split data into training, validation, and test sets with more samples in test set"""
        # First split: separate test set (10% of total data)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        # Second split: split remaining data into training and validation
        # Using 0.125 (1/8) of remaining data for validation to maintain similar validation size
        val_size = 0.125
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_size, 
            random_state=42, 
            stratify=y_train_val
        )
        
        self.logger.info(f"Data split sizes:")
        self.logger.info(f"Training: X={X_train.shape}, y={len(y_train)}")
        self.logger.info(f"Validation: X={X_val.shape}, y={len(y_val)}")
        self.logger.info(f"Test: X={X_test.shape}, y={len(y_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the model with proper validation"""
        print(f"Training model for {self.cancer_type}...")
        
        # Initialize model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        model = EnhancedCancerDetector(input_dim, num_classes)
        
        # Convert labels to one-hot encoding
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)
        
        # Calculate class weights
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.results_dir, f'best_model_{self.cancer_type}.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.train(
            X_train, y_train_onehot,
            X_val, y_val_onehot,
            batch_size=32,
            epochs=100,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        print(f"Evaluating {self.cancer_type} model...")
        
        # Convert labels to one-hot encoding
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=len(np.unique(y_test)))
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred_classes, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_classes)
        auc = roc_auc_score(y_test, y_pred[:, 1])
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.cancer_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Plot ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.cancer_type}')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, 'roc_curve.png'))
        plt.close()
        
        return report, cm, auc
    
    def cross_validate(self, X, y, n_splits=5):
        """Perform k-fold cross-validation"""
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            logging.info(f"\nTraining Fold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Apply data augmentation to training set
            X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
            
            # Train model
            model, history = self.train_model(X_train_aug, y_train_aug, X_val, y_val)
            
            # Evaluate model
            report, cm, auc = self.evaluate_model(model, X_val, y_val)
            
            # Store fold results
            fold_results = {
                'fold': fold,
                'report': report,
                'confusion_matrix': cm.tolist(),
                'auc': auc,
                'history': {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'accuracy': history.history['accuracy'],
                    'val_accuracy': history.history['val_accuracy']
                }
            }
            
            cv_scores.append(fold_results)
            
            # Log fold results
            logging.info(f"Fold {fold} Results:")
            logging.info(f"Accuracy: {report['accuracy']:.4f}")
            logging.info(f"Precision: {report['weighted avg']['precision']:.4f}")
            logging.info(f"Recall: {report['weighted avg']['recall']:.4f}")
            logging.info(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
            logging.info(f"AUC: {auc:.4f}")
        
        return cv_scores
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Perform cross-validation
        cv_scores = self.cross_validate(X, y)
        
        # Calculate average metrics
        avg_metrics = {
            'accuracy': np.mean([score['report']['accuracy'] for score in cv_scores]),
            'auc': np.mean([score['auc'] for score in cv_scores]),
            'precision': np.mean([score['report']['weighted avg']['precision'] for score in cv_scores]),
            'recall': np.mean([score['report']['weighted avg']['recall'] for score in cv_scores]),
            'f1-score': np.mean([score['report']['weighted avg']['f1-score'] for score in cv_scores])
        }
        
        # Save results
        results = {
            'cancer_type': self.cancer_type,
            'cv_scores': cv_scores,
            'avg_metrics': avg_metrics,
            'data_info': {
                'total_samples': len(y),
                'class_distribution': np.bincount(y).tolist(),
                'feature_count': X.shape[1]
            }
        }
        
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        return results

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cancer_type = 'BRCA'
    
    trainer = ModelTrainer(base_dir, cancer_type)
    results = trainer.run_pipeline()
    
    print("\nFinal Results:")
    print(f"Average Accuracy: {results['avg_metrics']['accuracy']:.3f}")
    print(f"Average AUC: {results['avg_metrics']['auc']:.3f}")
    print(f"Average F1-Score: {results['avg_metrics']['f1-score']:.3f}")

if __name__ == '__main__':
    main() 