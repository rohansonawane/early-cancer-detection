import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
from .snf_gcn import CrossOmicsGCN, SimilarityNetworkFusion
from utils.explainability import ModelExplainer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads=4, head_size=64, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        
    def build(self, input_shape):
        self.query_dense = layers.Dense(self.num_heads * self.head_size)
        self.key_dense = layers.Dense(self.num_heads * self.head_size)
        self.value_dense = layers.Dense(self.num_heads * self.head_size)
        self.combine_heads = layers.Dense(input_shape[-1])
        
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Add sequence dimension if not present
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)
        
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.num_heads * self.head_size))
        
        output = self.combine_heads(concat_attention)
        
        # Remove sequence dimension if it was added
        if len(inputs.shape) == 3:
            output = tf.squeeze(output, axis=1)
        
        return output

class ImprovedCancerDetector:
    def __init__(self, input_dim, num_classes, pathway_data=None):
        """
        Initialize the ImprovedCancerDetector model
        
        Args:
            input_dim (int): Input dimension (number of features)
            num_classes (int): Number of cancer subtypes to predict
            pathway_data (dict): Dictionary mapping pathways to genes
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pathway_data = pathway_data
        self.model = self._build_model()
        self.explainer = None  # Initialize explainer after model is built
        
    def _build_model(self):
        # Input layers for different omics data
        mrna_dim = self.input_dim // 3
        mirna_dim = self.input_dim // 3
        snv_dim = self.input_dim - mrna_dim - mirna_dim
        
        mrna_input = layers.Input(shape=(mrna_dim,), name='mrna_input')
        mirna_input = layers.Input(shape=(mirna_dim,), name='mirna_input')
        snv_input = layers.Input(shape=(snv_dim,), name='snv_input')
        
        # Process each omics data separately with feature-wise attention
        def create_omics_branch(inputs, name):
            # Feature-wise attention
            attention = MultiHeadAttention(num_heads=8, head_size=32)(inputs)
            x = layers.Add()([inputs, attention])  # Residual connection
            
            # Initial dense layer
            x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            
            # Residual block 1
            residual = x
            x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, residual])
            
            # Residual block 2
            residual = x
            x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, residual])
            
            return x
        
        mrna_branch = create_omics_branch(mrna_input, 'mrna')
        mirna_branch = create_omics_branch(mirna_input, 'mirna')
        snv_branch = create_omics_branch(snv_input, 'snv')
        
        # Concatenate omics data
        combined = layers.concatenate([mrna_branch, mirna_branch, snv_branch])
        
        # Multi-head attention layer with more heads
        attention_output = MultiHeadAttention(num_heads=8, head_size=64)(combined)
        
        # Additional processing with residual connections
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(attention_output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Residual block
        residual = x
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        
        # Final classification layers
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create model
        model = Model(
            inputs=[mrna_input, mirna_input, snv_input],
            outputs=outputs
        )
        
        # Compile model with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0  # Gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, callbacks=None, class_weight=None):
        """
        Train the model
        
        Args:
            X_train: Training data (tuple of mrna, mirna, snv)
            y_train: Training labels
            X_val: Validation data (tuple of mrna, mirna, snv)
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            callbacks: List of callbacks for training
            class_weight: Optional dictionary mapping class indices to weights
        """
        if callbacks is None:
            callbacks = []
        
        # Convert to numpy arrays
        X_train = X_train.values if hasattr(X_train, 'values') else X_train
        X_val = X_val.values if hasattr(X_val, 'values') else X_val
        
        # Split input data
        feature_dim = X_train.shape[1]
        mrna_dim = feature_dim // 3
        mirna_dim = feature_dim // 3
        snv_dim = feature_dim - mrna_dim - mirna_dim
        
        mrna_train = X_train[:, :mrna_dim]
        mirna_train = X_train[:, mrna_dim:mrna_dim + mirna_dim]
        snv_train = X_train[:, mrna_dim + mirna_dim:]
        
        mrna_val = X_val[:, :mrna_dim]
        mirna_val = X_val[:, mrna_dim:mrna_dim + mirna_dim]
        snv_val = X_val[:, mrna_dim + mirna_dim:]
        
        history = self.model.fit(
            [mrna_train, mirna_train, snv_train],
            y_train,
            validation_data=([mrna_val, mirna_val, snv_val], y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weight
        )
        return history
    
    def predict(self, X):
        """Make predictions on new data"""
        # Convert to numpy array
        X = X.values if hasattr(X, 'values') else X
        
        feature_dim = X.shape[1]
        mrna_dim = feature_dim // 3
        mirna_dim = feature_dim // 3
        snv_dim = feature_dim - mrna_dim - mirna_dim
        
        mrna_data = X[:, :mrna_dim]
        mirna_data = X[:, mrna_dim:mrna_dim + mirna_dim]
        snv_data = X[:, mrna_dim + mirna_dim:]
        
        return self.model.predict([mrna_data, mirna_data, snv_data])
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        # Convert to numpy array
        X_test = X_test.values if hasattr(X_test, 'values') else X_test
        
        feature_dim = X_test.shape[1]
        mrna_dim = feature_dim // 3
        mirna_dim = feature_dim // 3
        snv_dim = feature_dim - mrna_dim - mirna_dim
        
        mrna_test = X_test[:, :mrna_dim]
        mirna_test = X_test[:, mrna_dim:mrna_dim + mirna_dim]
        snv_test = X_test[:, mrna_dim + mirna_dim:]
        
        return self.model.evaluate([mrna_test, mirna_test, snv_test], y_test)
    
    def get_attention_weights(self, X):
        """Get attention weights for interpretation"""
        attention_layer = self.model.get_layer('multi_head_attention')
        attention_model = Model(
            inputs=self.model.inputs,
            outputs=attention_layer.output
        )
        mrna_data, mirna_data, snv_data = np.split(X, 3, axis=1)
        return attention_model.predict([mrna_data, mirna_data, snv_data])
    
    def explain_predictions(self, X, background_data=None):
        """
        Explain model predictions using SHAP values
        
        Args:
            X: Input data
            background_data: Background data for SHAP computation
            
        Returns:
            explainer: ModelExplainer instance
        """
        if self.explainer is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self.explainer = ModelExplainer(self.model, feature_names, self.pathway_data)
        return self.explainer.compute_shap_values(X, background_data)
    
    def analyze_pathways(self, X, top_n=100):
        """
        Analyze pathway enrichment
        
        Args:
            X: Input data
            top_n: Number of top features to consider
            
        Returns:
            enriched_pathways: DataFrame with enriched pathways
        """
        if self.explainer is None:
            self.explain_predictions(X)
        return self.explainer.analyze_pathways(X, top_n)
    
    def identify_biomarkers(self, X, expression_data, clinical_data, top_n=50):
        """
        Identify potential biomarkers
        
        Args:
            X: Input data
            expression_data: Gene expression data
            clinical_data: Clinical data
            top_n: Number of top features to consider
            
        Returns:
            biomarkers: DataFrame with potential biomarkers
        """
        if self.explainer is None:
            self.explain_predictions(X)
        return self.explainer.identify_biomarkers(X, expression_data, clinical_data, top_n)
    
    def get_feature_importance(self, X, method='integrated_gradients'):
        """Get feature importance scores"""
        if self.explainer is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self.explainer = ModelExplainer(self.model, feature_names, self.pathway_data)
        return self.explainer.explain_predictions(self.model, X, method=method) 