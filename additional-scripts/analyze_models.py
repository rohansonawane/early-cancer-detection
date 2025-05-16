import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import tensorflow as tf
from models.enhanced_cancer_detector import EnhancedMultiHeadAttention, ResidualBlock

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Alignment scores
        e = tf.matmul(x, self.W)
        
        # Get attention weights
        a = tf.nn.softmax(e, axis=1)
        
        # Apply attention weights
        output = x * tf.broadcast_to(a, tf.shape(x))
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape

# Define the custom MultiHeadAttention configuration
class CustomMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(num_heads=8, key_dim=64, **kwargs)
    
    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None):
        if key is None:
            key = value
        return super().call(
            query=query,
            value=value,
            key=key,
            attention_mask=attention_mask,
            return_attention_scores=return_attention_scores,
            training=training
        )

class ModelAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'analysis_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_data(self, cancer_type):
        """Load processed data for a specific cancer type"""
        processed_dir = os.path.join(self.base_dir, 'processed', cancer_type)
        mrna_data = pd.read_csv(os.path.join(processed_dir, 'mrna_processed.csv'), index_col=0)
        mirna_data = pd.read_csv(os.path.join(processed_dir, 'mirna_processed.csv'), index_col=0)
        snv_data = pd.read_csv(os.path.join(processed_dir, 'snv_processed.csv'), index_col=0)
        labels = pd.read_csv(os.path.join(processed_dir, 'labels.csv'), index_col=0)
        
        return mrna_data, mirna_data, snv_data, labels
    
    def load_model(self, cancer_type):
        """Load a trained model for a specific cancer type.
        
        Args:
            cancer_type (str): The type of cancer (e.g., 'BRCA', 'BLCA')
            
        Returns:
            tf.keras.Model: The loaded model with custom layers
        """
        model_path = os.path.join('models', f"{cancer_type}_model.h5")
        print(f"Loading model from {model_path}...")
        
        return tf.keras.models.load_model(
            model_path,
            custom_objects={
                'CustomMultiHeadAttention': CustomMultiHeadAttention,
                'PositionalEncoding': PositionalEncoding
            },
            compile=False
        )
    
    def plot_confusion_matrix(self, y_true, y_pred, cancer_type):
        """Plot confusion matrix"""
        cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {cancer_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.results_dir, f'{cancer_type}_confusion_matrix.png'))
        plt.close()
    
    def plot_roc_curves(self, y_true, y_pred, cancer_type):
        """Plot ROC curves"""
        n_classes = y_true.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {cancer_type}')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, f'{cancer_type}_roc_curves.png'))
        plt.close()
        
        return roc_auc
    
    def plot_precision_recall(self, y_true, y_pred, cancer_type):
        """Plot precision-recall curves"""
        n_classes = y_true.shape[1]
        precision = dict()
        recall = dict()
        avg_precision = dict()
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            avg_precision[i] = np.mean(precision[i])
            plt.plot(recall[i], precision[i], label=f'Class {i} (AP = {avg_precision[i]:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {cancer_type}')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, f'{cancer_type}_precision_recall.png'))
        plt.close()
        
        return avg_precision
    
    def analyze_feature_importance(self, model, X, cancer_type):
        """Analyze feature importance using attention weights."""
        # Find the attention layer by searching for layers that start with 'attention_layer'
        attention_layer_name = None
        for layer in model.layers:
            if layer.name.startswith('attention_layer'):
                attention_layer_name = layer.name
                break
        
        if attention_layer_name is None:
            raise ValueError(f"No attention layer found in {cancer_type} model")
            
        attention_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(attention_layer_name).output
        )
        attention_weights = attention_model.predict(X)
        
        # Calculate mean attention weights for each feature
        feature_importance = np.mean(attention_weights, axis=0)
        
        # Plot top 20 most important features
        plt.figure(figsize=(12, 6))
        sorted_idx = np.argsort(feature_importance)[-20:]
        plt.barh(range(20), feature_importance[sorted_idx])
        plt.title(f'Top 20 Most Important Features - {cancer_type}')
        plt.xlabel('Feature Importance Score')
        plt.savefig(os.path.join(self.results_dir, f'{cancer_type}_feature_importance.png'))
        plt.close()
        
        return feature_importance
    
    def generate_performance_report(self, cancer_type, y_true, y_pred, roc_auc, avg_precision):
        """Generate detailed performance report"""
        report = classification_report(
            np.argmax(y_true, axis=1),
            np.argmax(y_pred, axis=1),
            output_dict=True
        )
        
        # Add ROC AUC and Average Precision scores
        for i in range(len(roc_auc)):
            report[str(i)]['roc_auc'] = roc_auc[i]
            report[str(i)]['avg_precision'] = avg_precision[i]
        
        # Save report
        pd.DataFrame(report).to_csv(
            os.path.join(self.results_dir, f'{cancer_type}_performance_report.csv')
        )
        
        return report
    
    def analyze_cancer_type(self, cancer_type):
        """Analyze model performance for a specific cancer type"""
        print(f"\nAnalyzing {cancer_type} model...")
        
        # Load data and model
        mrna_data, mirna_data, snv_data, labels = self.load_data(cancer_type)
        X = pd.concat([mrna_data, mirna_data, snv_data], axis=1)
        model = self.load_model(cancer_type)
        
        # Get predictions
        y_pred = model.predict(X)
        y_true = tf.keras.utils.to_categorical(labels['response'])
        
        # Generate visualizations
        self.plot_confusion_matrix(y_true, y_pred, cancer_type)
        roc_auc = self.plot_roc_curves(y_true, y_pred, cancer_type)
        avg_precision = self.plot_precision_recall(y_true, y_pred, cancer_type)
        feature_importance = self.analyze_feature_importance(model, X, cancer_type)
        
        # Generate performance report
        report = self.generate_performance_report(cancer_type, y_true, y_pred, roc_auc, avg_precision)
        
        print(f"{cancer_type} analysis completed.")
        return report
    
    def analyze_all_models(self):
        """Analyze all cancer type models"""
        cancer_types = ['BLCA', 'LIHC', 'PRAD', 'BRCA', 'AML', 'WT']
        all_reports = {}
        
        for cancer_type in cancer_types:
            all_reports[cancer_type] = self.analyze_cancer_type(cancer_type)
        
        # Compare performance across cancer types
        self.plot_performance_comparison(all_reports)
        
        return all_reports
    
    def plot_performance_comparison(self, all_reports):
        """Plot performance comparison across cancer types"""
        metrics = ['precision', 'recall', 'f1-score']  # Removed 'accuracy' as it's not directly in the report
        cancer_types = list(all_reports.keys())
        
        plt.figure(figsize=(15, 10))
        x = np.arange(len(cancer_types))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [report['weighted avg'][metric] for report in all_reports.values()]
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Cancer Types')
        plt.ylabel('Score')
        plt.title('Performance Comparison Across Cancer Types')
        plt.xticks(x + width*1.5, cancer_types)
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, 'performance_comparison.png'))
        plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    analyzer = ModelAnalyzer(base_dir)
    all_reports = analyzer.analyze_all_models()
    
    # Print summary
    print("\nAnalysis Summary:")
    for cancer_type, report in all_reports.items():
        print(f"\n{cancer_type}:")
        print(f"Weighted Average Metrics:")
        for metric, value in report['weighted avg'].items():
            print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main() 