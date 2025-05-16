import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.layers import Input, Concatenate, Dense, LayerNormalization, Add

class EnhancedMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads=8, head_size=64, dropout_rate=0.1, **kwargs):
        super(EnhancedMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.query_dense = layers.Dense(self.num_heads * self.head_size)
        self.key_dense = layers.Dense(self.num_heads * self.head_size)
        self.value_dense = layers.Dense(self.num_heads * self.head_size)
        self.combine_heads = layers.Dense(input_shape[-1])
        self.dropout = layers.Dropout(self.dropout_rate)
        
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.dropout(weights)
        output = tf.matmul(weights, value)
        return output
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Linear projections
        q = self.query_dense(inputs)
        k = self.key_dense(inputs)
        v = self.value_dense(inputs)
        
        # Separate heads
        q = self.separate_heads(q, batch_size)
        k = self.separate_heads(k, batch_size)
        v = self.separate_heads(v, batch_size)
        
        # Attention
        attention_output = self.attention(q, k, v)
        
        # Combine heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, self.num_heads * self.head_size))
        
        # Final linear projection
        output = self.combine_heads(attention_output)
        return output

class ResidualBlock(layers.Layer):
    def __init__(self, units, dropout_rate=0.3, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.dense1 = layers.Dense(self.units, activation='relu', 
                                 kernel_regularizer=regularizers.l2(0.01))
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dense2 = layers.Dense(self.units, activation='relu',
                                 kernel_regularizer=regularizers.l2(0.01))
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(self.dropout_rate)
        
        # Projection layer if input shape doesn't match output shape
        if input_shape[-1] != self.units:
            self.projection = layers.Dense(self.units)
        else:
            self.projection = None
            
    def call(self, inputs):
        residual = inputs
        if self.projection:
            residual = self.projection(residual)
            
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        return layers.Add()([x, residual])

class EnhancedCancerDetector:
    def __init__(self, input_dim, num_classes, pathway_data=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pathway_data = pathway_data
        self.model = self._build_model()
        
    def _build_model(self):
        # Input layers for different omics data
        mrna_dim = self.input_dim // 3
        mirna_dim = self.input_dim // 3
        snv_dim = self.input_dim - mrna_dim - mirna_dim
        
        mrna_input = layers.Input(shape=(mrna_dim,), name='mrna_input')
        mirna_input = layers.Input(shape=(mirna_dim,), name='mirna_input')
        snv_input = layers.Input(shape=(snv_dim,), name='snv_input')
        
        def create_omics_branch(inputs, name):
            # Initial feature extraction
            x = layers.Dense(256, activation='relu', 
                           kernel_regularizer=regularizers.l2(0.01))(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            # Enhanced attention
            attention = EnhancedMultiHeadAttention(num_heads=8, head_size=32)(x)
            x = layers.Add()([x, attention])  # Residual connection
            
            # Residual blocks
            x = ResidualBlock(256)(x)
            x = ResidualBlock(256)(x)
            
            return x
        
        # Process each omics type
        mrna_branch = create_omics_branch(mrna_input, 'mrna')
        mirna_branch = create_omics_branch(mirna_input, 'mirna')
        snv_branch = create_omics_branch(snv_input, 'snv')
        
        # Concatenate omics data
        combined = layers.concatenate([mrna_branch, mirna_branch, snv_branch])
        
        # Global attention
        attention_output = EnhancedMultiHeadAttention(num_heads=8, head_size=64)(combined)
        x = layers.Add()([combined, attention_output])
        
        # Final processing
        x = ResidualBlock(512)(x)
        x = layers.Dense(256, activation='relu', 
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create model
        model = Model(
            inputs=[mrna_input, mirna_input, snv_input],
            outputs=outputs
        )
        
        # Compile model with improved optimizer settings
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', AUC(), Precision(), Recall()]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, callbacks=None, class_weight=None):
        if callbacks is None:
            callbacks = []
            
        # Add default callbacks if none provided
        if not callbacks:
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
                    filepath='best_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                )
            ]
        
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
        
        # Train model
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
        X_test = X_test.values if hasattr(X_test, 'values') else X_test
        feature_dim = X_test.shape[1]
        mrna_dim = feature_dim // 3
        mirna_dim = feature_dim // 3
        snv_dim = feature_dim - mrna_dim - mirna_dim
        
        mrna_test = X_test[:, :mrna_dim]
        mirna_test = X_test[:, mrna_dim:mrna_dim + mirna_dim]
        snv_test = X_test[:, mrna_dim + mirna_dim:]
        
        return self.model.evaluate([mrna_test, mirna_test, snv_test], y_test) 