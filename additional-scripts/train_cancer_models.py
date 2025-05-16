import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from models.cancer_detector import CancerDetector
import matplotlib.pyplot as plt
import seaborn as sns
from utils.process_cancer_data import CancerDataProcessor

class CancerModelTrainer:
    def __init__(self, base_dir):
        """
        Initialize the CancerModelTrainer
        
        Args:
            base_dir (str): Base directory containing the data
        """
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
    def train_cancer_model(self, cancer_type):
        """
        Train model for a specific cancer type
        
        Args:
            cancer_type (str): Type of cancer (e.g., 'BLCA', 'BRCA', etc.)
        """
        print(f"Training model for {cancer_type}...")
        
        # Load processed data
        processed_dir = os.path.join(self.base_dir, 'processed', cancer_type)
        mrna_data = pd.read_csv(os.path.join(processed_dir, 'mrna_processed.csv'), index_col=0)
        mirna_data = pd.read_csv(os.path.join(processed_dir, 'mirna_processed.csv'), index_col=0)
        snv_data = pd.read_csv(os.path.join(processed_dir, 'snv_processed.csv'), index_col=0)
        labels = pd.read_csv(os.path.join(processed_dir, 'labels.csv'), index_col=0)
        
        # Combine features
        X = pd.concat([mrna_data, mirna_data, snv_data], axis=1)
        y = labels['response'].values
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(le.classes_)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        
        # Initialize and train model
        model = CancerDetector(input_dim=X.shape[1], num_classes=num_classes)
        history = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"{cancer_type} Test accuracy: {test_acc:.4f}")
        print(f"{cancer_type} Test loss: {test_loss:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, f'{cancer_type}_model.h5')
        model.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Plot training history
        self._plot_training_history(history, cancer_type)
        
        return model, history, (X_test, y_test)
    
    def _plot_training_history(self, history, cancer_type):
        """Plot training history for a specific cancer type"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{cancer_type} Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{cancer_type} Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, f'{cancer_type}_training_history.png'))
        plt.close()
    
    def train_all_models(self):
        """Train models for all cancer types"""
        cancer_types = ['BLCA', 'LIHC', 'PRAD', 'BRCA', 'AML', 'WT']
        results = {}
        
        for cancer_type in cancer_types:
            model, history, (X_test, y_test) = self.train_cancer_model(cancer_type)
            results[cancer_type] = {
                'model': model,
                'history': history,
                'test_data': (X_test, y_test)
            }
        
        return results

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Set base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process data first
    processor = CancerDataProcessor(base_dir)
    processor.process_all_cancers()
    
    # Train models
    trainer = CancerModelTrainer(base_dir)
    results = trainer.train_all_models()

if __name__ == '__main__':
    main() 