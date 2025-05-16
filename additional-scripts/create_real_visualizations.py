import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def load_metrics(cancer_type):
    """Load metrics for a specific cancer type"""
    metrics_file = os.path.join('results', cancer_type, 'test_metrics.csv')
    if os.path.exists(metrics_file):
        return pd.read_csv(metrics_file)
    return None

def create_real_visualizations():
    # Create results directory if it doesn't exist
    if not os.path.exists('results/real_visualizations'):
        os.makedirs('results/real_visualizations')
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    # Cancer types
    cancer_types = ['AML', 'BRCA', 'PRAD', 'LIHC', 'BLCA']
    
    # Load metrics for all cancer types
    metrics_data = {}
    for cancer in cancer_types:
        metrics = load_metrics(cancer)
        if metrics is not None:
            metrics_data[cancer] = metrics.set_index('Metric')['Value'].to_dict()
    
    # 1. Performance Metrics Comparison
    def plot_performance_metrics():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1_score']
        x = np.arange(len(cancer_types))
        width = 0.15
        
        # Plot accuracy and AUC
        for i, metric in enumerate(['accuracy', 'auc']):
            values = [metrics_data[cancer].get(metric, 0) for cancer in cancer_types]
            ax1.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance by Cancer Type')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(cancer_types)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Plot precision, recall, and F1
        for i, metric in enumerate(['precision', 'recall', 'f1_score']):
            values = [metrics_data[cancer].get(metric, 0) for cancer in cancer_types]
            ax2.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax2.set_ylabel('Score')
        ax2.set_title('Detailed Metrics by Cancer Type')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(cancer_types)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/real_visualizations/performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Radar Plot of Metrics
    def plot_radar_metrics():
        # Number of metrics
        metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1_score']
        N = len(metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot each cancer type
        for cancer in cancer_types:
            values = [metrics_data[cancer].get(metric, 0) for metric in metrics]
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, label=cancer)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels
        plt.xticks(angles[:-1], metrics)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))
        plt.title('Performance Metrics Radar Plot')
        
        plt.tight_layout()
        plt.savefig('results/real_visualizations/radar_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Heatmap of Performance Metrics
    def plot_metrics_heatmap():
        # Create DataFrame of metrics
        metrics_df = pd.DataFrame(metrics_data).T
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', vmin=0, vmax=1)
        plt.title('Performance Metrics Heatmap')
        plt.tight_layout()
        plt.savefig('results/real_visualizations/metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Comparative Analysis
    def plot_comparative_analysis():
        # Calculate average performance
        metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1_score']
        avg_metrics = {}
        for metric in metrics:
            values = [metrics_data[cancer].get(metric, 0) for cancer in cancer_types]
            avg_metrics[metric] = np.mean(values)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual cancer types
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, cancer in enumerate(cancer_types):
            values = [metrics_data[cancer].get(metric, 0) for metric in metrics]
            ax.bar(x + i*width, values, width, label=cancer)
        
        # Plot average performance
        avg_values = [avg_metrics[metric] for metric in metrics]
        ax.plot(x + width*2.5, avg_values, 'k--', label='Average', linewidth=2)
        
        ax.set_ylabel('Score')
        ax.set_title('Comparative Analysis of Model Performance')
        ax.set_xticks(x + width*2.5)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/real_visualizations/comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate all visualizations
    plot_performance_metrics()
    plot_radar_metrics()
    plot_metrics_heatmap()
    plot_comparative_analysis()

if __name__ == '__main__':
    create_real_visualizations() 