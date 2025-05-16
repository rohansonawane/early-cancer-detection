import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import logging
from scipy import stats
import warnings

class DataPreprocessor:
    def __init__(self, config: Dict):
        """Initialize the data preprocessor.
        
        Args:
            config (Dict): Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.imputers = {}
        self.feature_selector = None
        self.pca = None
        self.label_encoder = LabelEncoder()
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for required columns and data types.
        
        Args:
            data (pd.DataFrame): Input data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            # Check required columns
            required_cols = self.config.get('required_columns', [])
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check data types
            for col, dtype in self.config.get('column_types', {}).items():
                if col in data.columns and not pd.api.types.is_dtype_equal(data[col].dtype, dtype):
                    raise ValueError(f"Invalid data type for column {col}")
            
            return True
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return False
            
    def handle_missing_values(self, data: pd.DataFrame, method: str = 'knn') -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Imputation method ('knn' or 'simple')
            
        Returns:
            pd.DataFrame: Data with imputed values
        """
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # Handle numeric missing values
            if method == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            else:
                imputer = SimpleImputer(strategy='median')
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            
            # Handle categorical missing values
            for col in categorical_cols:
                data[col] = data[col].fillna(data[col].mode()[0])
            
            return data
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise
            
    def remove_outliers(self, data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Remove outliers from the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Outlier detection method ('zscore' or 'iqr')
            
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if method == 'zscore':
                z_scores = stats.zscore(data[numeric_cols])
                data = data[(np.abs(z_scores) < 3).all(axis=1)]
            else:  # IQR method
                for col in numeric_cols:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]
            
            return data
        except Exception as e:
            self.logger.error(f"Error removing outliers: {str(e)}")
            raise
            
    def normalize_features(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Normalize features in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Normalization method ('standard' or 'robust')
            
        Returns:
            pd.DataFrame: Normalized data
        """
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if method == 'standard':
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
                
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            self.scalers[method] = scaler
            
            return data
        except Exception as e:
            self.logger.error(f"Error normalizing features: {str(e)}")
            raise
            
    def select_features(self, data: pd.DataFrame, target: str, method: str = 'mutual_info', k: int = 100) -> pd.DataFrame:
        """Select most important features.
        
        Args:
            data (pd.DataFrame): Input data
            target (str): Target column name
            method (str): Feature selection method ('mutual_info' or 'f_classif')
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Data with selected features
        """
        try:
            X = data.drop(target, axis=1)
            y = data[target]
            
            if method == 'mutual_info':
                selector = SelectKBest(mutual_info_classif, k=k)
            else:
                selector = SelectKBest(f_classif, k=k)
                
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            self.feature_selector = selector
            return data[selected_features + [target]]
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            raise
            
    def apply_pca(self, data: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
        """Apply Principal Component Analysis.
        
        Args:
            data (pd.DataFrame): Input data
            n_components (int): Number of components to keep
            
        Returns:
            pd.DataFrame: Transformed data
        """
        try:
            self.pca = PCA(n_components=n_components)
            transformed_data = self.pca.fit_transform(data)
            
            # Create column names for PCA components
            pca_columns = [f'PC{i+1}' for i in range(n_components)]
            return pd.DataFrame(transformed_data, columns=pca_columns)
        except Exception as e:
            self.logger.error(f"Error applying PCA: {str(e)}")
            raise
            
    def encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Encoded data
        """
        try:
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                data[col] = self.label_encoder.fit_transform(data[col])
            
            return data
        except Exception as e:
            self.logger.error(f"Error encoding categorical variables: {str(e)}")
            raise
            
    def preprocess(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Complete preprocessing pipeline.
        
        Args:
            train_data (pd.DataFrame): Training data
            test_data (pd.DataFrame): Testing data
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Preprocessed data
        """
        try:
            # Validate data
            if not self.validate_data(train_data) or not self.validate_data(test_data):
                raise ValueError("Data validation failed")
            
            # Handle missing values
            train_data = self.handle_missing_values(train_data)
            test_data = self.handle_missing_values(test_data)
            
            # Remove outliers from training data
            train_data = self.remove_outliers(train_data)
            
            # Encode categorical variables
            train_data = self.encode_categorical(train_data)
            test_data = self.encode_categorical(test_data)
            
            # Normalize features
            train_data = self.normalize_features(train_data)
            test_data = self.normalize_features(test_data)
            
            # Select features
            target_col = self.config['target_column']
            train_data = self.select_features(train_data, target_col)
            test_data = test_data[train_data.columns]
            
            # Apply PCA if specified
            if self.config.get('use_pca', False):
                train_data = self.apply_pca(train_data)
                test_data = self.apply_pca(test_data)
            
            # Split features and target
            X_train = train_data.drop(target_col, axis=1).values
            y_train = train_data[target_col].values
            X_test = test_data.drop(target_col, axis=1).values
            y_test = test_data[target_col].values
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise 