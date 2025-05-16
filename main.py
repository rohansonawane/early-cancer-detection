import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils.data_processor import MultiOmicsProcessor
from models.cancer_detector import CancerDetector
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

def load_data(data_dir):
    """Load multi-omics data"""
    data = {}
    
    # Try to load each data type
    try:
        data['mrna'] = pd.read_csv(os.path.join(data_dir, 'mRNA_data.csv'), index_col=0)
        print(f"Loaded mRNA data with shape: {data['mrna'].shape}")
    except FileNotFoundError:
        print("mRNA data not found")
        data['mrna'] = None
        
    try:
        data['mirna'] = pd.read_csv(os.path.join(data_dir, 'miRNA_data.csv'), index_col=0)
        print(f"Loaded miRNA data with shape: {data['mirna'].shape}")
    except FileNotFoundError:
        print("miRNA data not found")
        data['mirna'] = None
        
    try:
        data['snv'] = pd.read_csv(os.path.join(data_dir, 'snv_data.csv'), index_col=0)
        print(f"Loaded SNV data with shape: {data['snv'].shape}")
    except FileNotFoundError:
        print("SNV data not found")
        data['snv'] = None
    
    try:
        data['labels'] = pd.read_csv(os.path.join(data_dir, 'response.csv'), index_col=0)
        print(f"Loaded response data with shape: {data['labels'].shape}")
    except FileNotFoundError:
        raise FileNotFoundError("Response data is required but not found")
    
    return data

def preprocess_data(data, kegg_pathways_file):
    """Preprocess and integrate multi-omics data"""
    processor = MultiOmicsProcessor(kegg_pathways_file)
    
    # Process available data types
    processed_data = {}
    
    if data['mrna'] is not None:
        processed_data['mrna'] = processor.process_mrna_data(data['mrna'])
    
    if data['mirna'] is not None:
        processed_data['mirna'] = processor.process_mirna_data(data['mirna'])
    
    if data['snv'] is not None:
        processed_data['snv'] = processor.process_snv_data(data['snv'])
    
    # Create pathway features if both mRNA and miRNA are available
    if data['mrna'] is not None and data['mirna'] is not None:
        pathway_features = processor.create_pathway_features(
            processed_data['mrna'], 
            processed_data['mirna']
        )
        processed_data['pathway'] = pathway_features
    
    # Combine all available features
    integrated_data = pd.concat([df for df in processed_data.values()], axis=1)
    
    # Scale features
    scaler = StandardScaler()
    integrated_data = pd.DataFrame(
        scaler.fit_transform(integrated_data),
        columns=integrated_data.columns,
        index=integrated_data.index
    )
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(data['labels'].values.ravel())
    num_classes = len(le.classes_)
    
    # Convert labels to one-hot encoding
    one_hot_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes)
    
    return integrated_data, one_hot_labels, num_classes, scaler

def train_model(X, y, num_classes, cancer_type, n_splits=5):
    """Train the cancer detection model with k-fold cross-validation"""
    # Initialize k-fold cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    models = []
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, np.argmax(y, axis=1)), 1):
        print(f"\nTraining Fold {fold}/{n_splits}")
        
    # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize model
        model = CancerDetector(
            input_dim=X.shape[1],
            num_classes=num_classes
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='min'
            ),
            ModelCheckpoint(
                filepath=f'results/{cancer_type}/best_model_fold_{fold}.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]
        
        # Train model
        history_main, history_secondary = model.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=32,
            epochs=100,
            callbacks=callbacks
        )
        
        # Get predictions
        y_pred = model.predict(X_val)
        
        # Convert predictions to numpy arrays
        y_pred = np.array(y_pred)
        y_val_np = np.array(y_val)
        
        # Ensure predictions have the correct shape
        if len(y_pred.shape) == 1:
            y_pred = np.expand_dims(y_pred, axis=1)
            y_pred = np.concatenate([1 - y_pred, y_pred], axis=1)
    
        # Ensure predictions have the same number of samples
        if y_pred.shape[0] != y_val_np.shape[0]:
            print(f"Warning: Prediction shape mismatch. y_pred: {y_pred.shape}, y_val: {y_val_np.shape}")
            y_pred = y_pred[:y_val_np.shape[0]]
        
        # Evaluate model
        val_metrics = {
            'accuracy': accuracy_score(np.argmax(y_val_np, axis=1), np.argmax(y_pred, axis=1)),
            'precision': precision_score(np.argmax(y_val_np, axis=1), np.argmax(y_pred, axis=1), average='weighted'),
            'recall': recall_score(np.argmax(y_val_np, axis=1), np.argmax(y_pred, axis=1), average='weighted'),
            'f1': f1_score(np.argmax(y_val_np, axis=1), np.argmax(y_pred, axis=1), average='weighted')
        }
        
        # Calculate AUC only for binary classification
        if y_pred.shape[1] == 2:
            val_metrics['auc'] = roc_auc_score(np.argmax(y_val_np, axis=1), y_pred[:, 1])
        else:
            val_metrics['auc'] = None
        
        # Combine histories from both models
        combined_history = {
            'main_model': history_main.history,
            'secondary_model': history_secondary.history
        }
        
        fold_results.append({
            'fold': fold,
            'metrics': val_metrics,
            'history': combined_history,
            'y_true': y_val_np,
            'y_pred': y_pred
        })
        
        models.append(model)
    
    # Calculate average metrics across folds
    avg_metrics = calculate_average_metrics(fold_results)
    
    # Save results
    save_results(cancer_type, fold_results, avg_metrics)
    
    return models, fold_results, avg_metrics

def evaluate_model(model, X, y):
    """Evaluate model performance with multiple metrics"""
    # Get predictions from the ensemble
    y_pred = model.predict(X)
    
    # Convert y to numpy array if it's not already
    if hasattr(y, 'values'):
        y = y.values
    
    # Get predicted classes from ensemble predictions
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y, axis=1)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_classes),
        'precision': precision_score(y_true, y_pred_classes, average='weighted'),
        'recall': recall_score(y_true, y_pred_classes, average='weighted'),
        'f1': f1_score(y_true, y_pred_classes, average='weighted')
    }
    
    # Calculate AUC only for binary classification
    if y_pred.shape[1] == 2:
        metrics['auc'] = roc_auc_score(y_true, y_pred[:, 1])
    else:
        metrics['auc'] = None
    
    return metrics

def calculate_average_metrics(fold_results):
    """Calculate average metrics across all folds"""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    avg_metrics = {}
    
    for metric in metrics:
        values = [fold['metrics'][metric] for fold in fold_results if fold['metrics'][metric] is not None]
        if values:
            avg_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    return avg_metrics

def save_results(cancer_type, fold_results, avg_metrics):
    """Save training results and metrics"""
    results_dir = os.path.join('results', cancer_type)
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert NumPy arrays and types to Python native types
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    # Convert fold results
    serializable_fold_results = []
    for fold_result in fold_results:
        serializable_fold = {
            'fold': fold_result['fold'],
            'metrics': convert_to_native(fold_result['metrics']),
            'history': convert_to_native(fold_result['history']),
            'y_true': convert_to_native(fold_result['y_true']),
            'y_pred': convert_to_native(fold_result['y_pred'])
        }
        serializable_fold_results.append(serializable_fold)
    
    # Convert average metrics
    serializable_avg_metrics = convert_to_native(avg_metrics)
    
    # Save fold results
    with open(os.path.join(results_dir, 'fold_results.json'), 'w') as f:
        json.dump(serializable_fold_results, f, indent=2)
    
    # Save average metrics
    with open(os.path.join(results_dir, 'average_metrics.json'), 'w') as f:
        json.dump(serializable_avg_metrics, f, indent=2)
    
    # Plot training curves
    plot_training_curves(fold_results, results_dir)
    
    # Plot confusion matrix for each fold
    plot_confusion_matrices(fold_results, results_dir)

def plot_training_curves(fold_results, results_dir):
    """Plot training and validation curves for each fold"""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    for fold in fold_results:
        plt.plot(fold['history']['main_model']['loss'], label=f'Training Loss (Fold {fold["fold"]})', alpha=0.3)
        plt.plot(fold['history']['main_model']['val_loss'], label=f'Validation Loss (Fold {fold["fold"]})', alpha=0.3)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 1, 2)
    for fold in fold_results:
        plt.plot(fold['history']['main_model']['accuracy'], label=f'Training Accuracy (Fold {fold["fold"]})', alpha=0.3)
        plt.plot(fold['history']['main_model']['val_accuracy'], label=f'Validation Accuracy (Fold {fold["fold"]})', alpha=0.3)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'))
    plt.close()
    
def plot_confusion_matrices(fold_results, results_dir):
    """Plot confusion matrices for each fold"""
    n_folds = len(fold_results)
    fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 5))
    
    for i, fold in enumerate(fold_results):
        cm = confusion_matrix(
            np.argmax(fold['y_true'], axis=1),
            np.argmax(fold['y_pred'], axis=1)
        )
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
        axes[i].set_title(f'Fold {i+1} Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'))
    plt.close()

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create main results directory
    os.makedirs('results', exist_ok=True)
    
    # List of cancer types
    cancer_types = ['WT', 'AML', 'BRCA', 'PRAD', 'LIHC', 'BLCA']
    
    # Process each cancer type
    for cancer_type in cancer_types:
        print(f'\nProcessing {cancer_type} dataset...')
        
    # Load data
        data_dir = f'data/{cancer_type}_data'
        try:
            data = load_data(data_dir)
        except FileNotFoundError as e:
            print(f'Error loading {cancer_type} data: {str(e)}')
            continue
    
    # Preprocess data
    kegg_pathways_file = 'KEGG_pathways/20230205_kegg_hsa.gmt'
        try:
            X, y, num_classes, scaler = preprocess_data(data, kegg_pathways_file)
            print(f'Processed data shape: {X.shape}')
        except Exception as e:
            print(f'Error preprocessing {cancer_type} data: {str(e)}')
            continue
        
        # Train model with cross-validation
        models, fold_results, avg_metrics = train_model(X, y, num_classes, cancer_type)
    
        # Print average metrics
        print(f"\nAverage metrics for {cancer_type}:")
        for metric, values in avg_metrics.items():
            print(f"{metric}: {values['mean']:.3f} ± {values['std']:.3f}")
        
        print(f'Completed processing {cancer_type}')

if __name__ == '__main__':
    main() 