# Early Detection of Cancer Subtypes Using Multi-Omics Data Integration and Explainable AI

This project implements a sophisticated deep learning system for early detection and classification of multiple cancer subtypes using multi-omics data integration. The system leverages KEGG pathway information and explainable AI techniques to provide interpretable predictions.

## Project Description

The project aims to revolutionize cancer detection by combining multiple types of biological data (multi-omics) with advanced deep learning techniques. By integrating gene expression, microRNA, and genetic variation data, the system can identify cancer subtypes at early stages, potentially improving patient outcomes through earlier intervention.

## Overview

The system is designed to detect and classify five major cancer types:
- Acute Myeloid Leukemia (AML): A type of blood cancer affecting white blood cells
- Breast Cancer (BRCA): The most common cancer in women worldwide
- Prostate Cancer (PRAD): A common cancer in men affecting the prostate gland
- Liver Cancer (LIHC): A serious cancer affecting liver function
- Bladder Cancer (BLCA): A cancer affecting the bladder's inner lining

## Features

### Data Integration
- Multi-omics data integration (gene expression, methylation, mutation)
  - Gene expression (mRNA): Measures the activity levels of genes
  - MicroRNA (miRNA): Small non-coding RNAs that regulate gene expression
  - Single Nucleotide Variations (SNV): Genetic variations at single nucleotide positions
- KEGG pathway-based feature engineering: Utilizes biological pathway information
- Cross-validation for robust performance evaluation
- Comprehensive performance metrics
- Explainable AI for model interpretability
- Visualization tools for results analysis

### Model Architecture
The system employs an ensemble approach with three main components:

1. **Main Deep Learning Model**:
   - Multi-head attention mechanism (8 heads): Captures complex relationships
   - Residual blocks with batch normalization: Ensures stable training
   - Feature-wise attention layers: Identifies important features
   - Dropout regularization (0.3-0.4): Prevents overfitting
   - L2 regularization (0.01): Controls model complexity

2. **Secondary Model**:
   - Convolutional-like processing: Captures local patterns
   - Global average pooling: Reduces dimensionality
   - Dense layers with batch normalization
   - Label smoothing (0.1): Improves generalization

3. **Random Forest Classifier**:
   - 100 estimators: Provides diverse predictions
   - Maximum depth of 10: Balances complexity
   - Used for ensemble diversity

### Training Configuration
- Batch size: 32 (Balances memory usage and training speed)
- Epochs: 100 (Maximum training iterations)
- Learning rate: 0.001 (Step size for optimization)
- Early stopping patience: 15 (Prevents overfitting)
- Learning rate reduction patience: 5 (Adapts to training progress)
- Cross-validation folds: 5 (Ensures robust evaluation)

## Performance Metrics

The model demonstrates strong performance across all cancer types:

| Cancer Type | Accuracy | Precision | Recall | F1 Score | AUC |
|-------------|----------|-----------|---------|-----------|-----|
| AML         | 0.751    | 0.752     | 0.751   | 0.751     | 0.843|
| BRCA        | 0.815    | 0.864     | 0.815   | 0.789     | 0.934|
| PRAD        | 0.716    | 0.736     | 0.716   | 0.686     | 0.783|
| LIHC        | 0.828    | 0.833     | 0.828   | 0.826     | 0.896|
| BLCA        | 0.818    | 0.826     | 0.818   | 0.809     | 0.916|

### Key Achievements
- High accuracy rates (71.6% - 81.5%): Demonstrates reliable classification
- Strong precision (73.6% - 86.4%): Shows low false positive rates
- Good recall rates (71.6% - 81.8%): Indicates effective detection
- Excellent AUC scores (78.3% - 93.4%): Reflects strong overall performance

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cancer-detection.git
cd cancer-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
cancer-detection/
├── data/                  # Data directory
│   ├── raw/              # Raw multi-omics data
│   └── processed/        # Processed data files
├── models/               # Model architecture definitions
├── results/              # Training results and visualizations
│   ├── AML/             # Results for each cancer type
│   ├── BRCA/
│   ├── PRAD/
│   ├── LIHC/
│   └── BLCA/
├── src/                  # Source code
│   ├── data/            # Data processing scripts
│   ├── models/          # Model training scripts
│   └── utils/           # Utility functions
├── notebooks/           # Jupyter notebooks for analysis
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Usage

1. Data Preparation:
```bash
python src/data/prepare_data.py
```

2. Model Training:
```bash
python src/models/train_model.py --cancer_type AML
```

3. Generate Results Visualization:
```bash
python create_results_poster.py
```

## Implementation Details

### Data Processing Pipeline
1. Data loading and preprocessing: Ensures data quality
2. Feature engineering using KEGG pathways: Enhances biological relevance
3. Data normalization and augmentation: Improves training stability
4. Cross-validation split: Ensures robust evaluation

### Model Training Pipeline
1. Ensemble model initialization: Sets up the learning framework
2. Cross-validation training: Ensures robust performance
3. Model evaluation and metrics calculation: Measures success
4. Results visualization and storage: Documents outcomes

### Evaluation Pipeline
1. Performance metrics calculation: Quantifies model success
2. Confusion matrix generation: Analyzes prediction patterns
3. ROC curve analysis: Evaluates discriminative ability
4. Statistical significance testing: Validates results

## Explainable AI Components

### Model Interpretability
- Multi-head attention visualization: Shows which features the model focuses on
- Feature importance analysis: Identifies key biomarkers for each cancer type
- Pathway enrichment analysis: Reveals biological mechanisms
- Confusion matrix analysis: Provides detailed error analysis

### Visualization Tools
- Performance metrics heatmap: Shows performance across cancer types
- Training and validation curves: Displays learning progress
- Confusion matrices: Visualizes prediction patterns
- Radar plots for metric comparison: Compares different aspects of performance

## Future Improvements

### Technical Enhancements
1. Integration of additional omics data types: Expand biological coverage
2. Advanced feature selection methods: Improve efficiency
3. Improved ensemble weighting strategies: Enhance performance
4. Enhanced explainability techniques: Increase transparency

### Clinical Applications
1. Real-time prediction system: Enable immediate analysis
2. Clinical decision support integration: Assist medical professionals
3. Patient-specific treatment recommendations: Enable personalized medicine
4. Longitudinal monitoring capabilities: Track disease progression

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```
@article{your-paper,
  title={Early Detection of Cancer Subtypes Using Multi-Omics Data Integration and Explainable AI},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Contact

For questions and feedback, please open an issue in the repository. 