import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class VisualizationManager:
    def __init__(self, style: str = 'seaborn'):
        """Initialize the visualization manager.
        
        Args:
            style (str): Plotting style ('seaborn' or 'default')
        """
        self.logger = logging.getLogger(__name__)
        plt.style.use(style)
        self.colors = sns.color_palette('husl', 8)
        
    def plot_learning_curves(self, history: Dict, save_path: Optional[str] = None):
        """Plot learning curves for training and validation metrics.
        
        Args:
            history (Dict): Training history
            save_path (Optional[str]): Path to save the plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot loss
            ax1.plot(history['loss'], label='Training Loss', color=self.colors[0])
            ax1.plot(history['val_loss'], label='Validation Loss', color=self.colors[1])
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracy
            ax2.plot(history['accuracy'], label='Training Accuracy', color=self.colors[2])
            ax2.plot(history['val_accuracy'], label='Validation Accuracy', color=self.colors[3])
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting learning curves: {str(e)}")
            raise
            
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot confusion matrix with annotations.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (Optional[List[str]]): Label names
            save_path (Optional[str]): Path to save the plot
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
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
                roc_auc = auc(fpr, tpr)
                
                label = labels[i] if labels else f'Class {i}'
                plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})',
                        color=self.colors[i % len(self.colors)])
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.grid(True)
            
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
                
                label = labels[i] if labels else f'Class {i}'
                plt.plot(recall, precision, label=label,
                        color=self.colors[i % len(self.colors)])
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting precision-recall curves: {str(e)}")
            raise
            
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                              top_n: int = 20, save_path: Optional[str] = None):
        """Plot feature importance scores.
        
        Args:
            feature_names (List[str]): Feature names
            importance_scores (np.ndarray): Feature importance scores
            top_n (int): Number of top features to plot
            save_path (Optional[str]): Path to save the plot
        """
        try:
            # Sort features by importance
            indices = np.argsort(importance_scores)[-top_n:]
            
            plt.figure(figsize=(12, 6))
            plt.barh(range(top_n), importance_scores[indices], color=self.colors[0])
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
            plt.xlabel('Importance Score')
            plt.title(f'Top {top_n} Feature Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")
            raise
            
    def plot_performance_metrics(self, metrics: Dict, save_path: Optional[str] = None):
        """Plot performance metrics.
        
        Args:
            metrics (Dict): Dictionary of performance metrics
            save_path (Optional[str]): Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 6))
            
            metrics_names = list(metrics.keys())
            metrics_values = list(metrics.values())
            
            plt.bar(metrics_names, metrics_values, color=self.colors)
            plt.xticks(rotation=45)
            plt.title('Model Performance Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {str(e)}")
            raise
            
    def create_interactive_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray, labels: Optional[List[str]] = None,
                               save_path: Optional[str] = None):
        """Create interactive plots using Plotly.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
            labels (Optional[List[str]]): Label names
            save_path (Optional[str]): Path to save the plot
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('ROC Curves', 'Precision-Recall Curves',
                              'Confusion Matrix', 'Performance Metrics')
            )
            
            # ROC Curves
            for i in range(y_pred_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                label = labels[i] if labels else f'Class {i}'
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=f'{label} (AUC = {roc_auc:.2f})'),
                    row=1, col=1
                )
            
            # Precision-Recall Curves
            for i in range(y_pred_proba.shape[1]):
                precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
                
                label = labels[i] if labels else f'Class {i}'
                fig.add_trace(
                    go.Scatter(x=recall, y=precision, name=label),
                    row=1, col=2
                )
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig.add_trace(
                go.Heatmap(z=cm, x=labels, y=labels, colorscale='Blues'),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                title_text='Interactive Model Performance Visualization',
                showlegend=True
            )
            
            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
                
        except Exception as e:
            self.logger.error(f"Error creating interactive plots: {str(e)}")
            raise
            
    def save_all_plots(self, plots: Dict[str, plt.Figure], save_dir: str):
        """Save all plots to files.
        
        Args:
            plots (Dict[str, plt.Figure]): Dictionary of plot names and figures
            save_dir (str): Directory to save plots
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for name, fig in plots.items():
                save_path = os.path.join(save_dir, f'{name}_{timestamp}.png')
                fig.savefig(save_path)
                plt.close(fig)
            
            self.logger.info(f"All plots saved to {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving plots: {str(e)}")
            raise

# Generate all visualizations
if __name__ == "__main__":
    # Create an instance of VisualizationManager
    vm = VisualizationManager()

    # Sample data (replace with your actual data)
    cancer_types = ['WT', 'AML', 'BRCA', 'PRAD', 'LIHC', 'BLCA']
    n_classes = len(cancer_types)

    # 1. Confusion Matrix
    vm.plot_confusion_matrix(np.array([0, 1, 0, 1, 0, 1]), np.array([0, 1, 0, 1, 0, 1]), labels=cancer_types)

    # 2. ROC Curves
    vm.plot_roc_curves(np.array([0, 1, 0, 1, 0, 1]), np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7]]), labels=cancer_types)

    # 3. Learning Curves
    vm.plot_learning_curves({'loss': [0.2, 0.15, 0.1, 0.05], 'val_loss': [0.25, 0.18, 0.12, 0.08], 'accuracy': [0.9, 0.92, 0.94, 0.96], 'val_accuracy': [0.88, 0.9, 0.92, 0.94]})

    # 4. Feature Importance
    vm.plot_feature_importance(['TP53', 'miR-21', 'BRCA1', 'FLT3', 'AR', 'CTNNB1', 'miR-155', 'PTEN', 'RB1', 'NPM1'], np.array([0.85, 0.82, 0.78, 0.75, 0.72, 0.70, 0.68, 0.65, 0.63, 0.60]))

    # 5. Performance Metrics
    vm.plot_performance_metrics({'Accuracy': 0.925, 'Precision': 0.918, 'Recall': 0.923, 'F1-Score': 0.920})

    # Create interactive plots
    vm.create_interactive_plots(np.array([0, 1, 0, 1, 0, 1]), np.array([0, 1, 0, 1, 0, 1]), np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7]]), labels=cancer_types)

    # Save all plots
    vm.save_all_plots({'Confusion Matrix': plt.gcf(), 'ROC Curves': plt.gcf(), 'Learning Curves': plt.gcf(), 'Feature Importance': plt.gcf(), 'Performance Metrics': plt.gcf()}, 'plots')

    print("All visualizations have been generated successfully!") 