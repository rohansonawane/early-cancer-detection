import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import json
import os
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        """Initialize the model evaluator."""
        self.logger = logging.getLogger(__name__)
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate various evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (Optional[np.ndarray]): Predicted probabilities
            
        Returns:
            Dict: Dictionary of metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            
            if y_pred_proba is not None:
                metrics.update({
                    'roc_auc': roc_auc_score(y_true, y_pred_proba, multi_class='ovr'),
                    'average_precision': average_precision_score(y_true, y_pred_proba)
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise
            
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                labels: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """Generate confusion matrix and related metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (Optional[List[str]]): Label names
            
        Returns:
            Tuple[np.ndarray, Dict]: Confusion matrix and metrics
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate per-class metrics
            class_metrics = {}
            for i in range(len(cm)):
                tp = cm[i, i]
                fp = sum(cm[:, i]) - tp
                fn = sum(cm[i, :]) - tp
                tn = sum(sum(cm)) - tp - fp - fn
                
                class_metrics[f'class_{i}'] = {
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                }
            
            return cm, class_metrics
            
        except Exception as e:
            self.logger.error(f"Error generating confusion matrix: {str(e)}")
            raise
            
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       labels: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot ROC curves for each class.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            labels (Optional[List[str]]): Label names
            save_path (Optional[str]): Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 8))
            
            for i in range(y_pred_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
                roc_auc = roc_auc_score(y_true == i, y_pred_proba[:, i])
                
                label = labels[i] if labels else f'Class {i}'
                plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting ROC curves: {str(e)}")
            raise
            
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   labels: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot precision-recall curves for each class.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            labels (Optional[List[str]]): Label names
            save_path (Optional[str]): Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 8))
            
            for i in range(y_pred_proba.shape[1]):
                precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
                avg_precision = average_precision_score(y_true == i, y_pred_proba[:, i])
                
                label = labels[i] if labels else f'Class {i}'
                plt.plot(recall, precision, label=f'{label} (AP = {avg_precision:.2f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting precision-recall curves: {str(e)}")
            raise
            
    def plot_confusion_matrix(self, cm: np.ndarray, labels: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
        """Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            labels (Optional[List[str]]): Label names
            save_path (Optional[str]): Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
            
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     labels: Optional[List[str]] = None) -> str:
        """Generate detailed classification report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (Optional[List[str]]): Label names
            
        Returns:
            str: Classification report
        """
        try:
            return classification_report(y_true, y_pred, target_names=labels)
            
        except Exception as e:
            self.logger.error(f"Error generating classification report: {str(e)}")
            raise
            
    def perform_statistical_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """Perform statistical analysis of predictions.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (Optional[np.ndarray]): Predicted probabilities
            
        Returns:
            Dict: Statistical analysis results
        """
        try:
            analysis = {
                'class_distribution': {
                    'true': np.bincount(y_true) / len(y_true),
                    'predicted': np.bincount(y_pred) / len(y_pred)
                },
                'error_analysis': {
                    'misclassification_rate': 1 - accuracy_score(y_true, y_pred),
                    'class_error_rates': {}
                }
            }
            
            # Calculate per-class error rates
            for i in range(len(np.unique(y_true))):
                class_mask = y_true == i
                class_errors = y_pred[class_mask] != i
                analysis['error_analysis']['class_error_rates'][f'class_{i}'] = np.mean(class_errors)
            
            # Add confidence analysis if probabilities are provided
            if y_pred_proba is not None:
                analysis['confidence_analysis'] = {
                    'mean_confidence': np.mean(np.max(y_pred_proba, axis=1)),
                    'confidence_std': np.std(np.max(y_pred_proba, axis=1)),
                    'confidence_by_class': {
                        f'class_{i}': np.mean(y_pred_proba[y_true == i, i])
                        for i in range(y_pred_proba.shape[1])
                    }
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error performing statistical analysis: {str(e)}")
            raise
            
    def save_evaluation_results(self, results: Dict, save_dir: str):
        """Save evaluation results to files.
        
        Args:
            results (Dict): Evaluation results
            save_dir (str): Directory to save results
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save metrics
            metrics_path = os.path.join(save_dir, f'metrics_{timestamp}.json')
            with open(metrics_path, 'w') as f:
                json.dump(results['metrics'], f, indent=4)
            
            # Save classification report
            report_path = os.path.join(save_dir, f'classification_report_{timestamp}.txt')
            with open(report_path, 'w') as f:
                f.write(results['classification_report'])
            
            # Save statistical analysis
            analysis_path = os.path.join(save_dir, f'statistical_analysis_{timestamp}.json')
            with open(analysis_path, 'w') as f:
                json.dump(results['statistical_analysis'], f, indent=4)
            
            self.logger.info(f"Evaluation results saved to {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {str(e)}")
            raise 