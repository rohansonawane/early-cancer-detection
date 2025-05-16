import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import tensorflow_addons as tfa

class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads=4, key_dim=64, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=0.1
        )
        
    def call(self, x):
        # Reshape input for attention
        x_reshaped = tf.expand_dims(x, axis=1)
        attention_output = self.attention(x_reshaped, x_reshaped)
        return tf.squeeze(attention_output, axis=1)

class ResidualBlock(layers.Layer):
    def __init__(self, units, dropout_rate=0.3, l2_reg=0.01, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
    def build(self, input_shape):
        self.dense1 = layers.Dense(
            self.units,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(self.dropout_rate)
        
        self.dense2 = layers.Dense(
            self.units,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(self.dropout_rate)
        
        if input_shape[-1] != self.units:
            self.projection = layers.Dense(
                self.units,
                kernel_regularizer=regularizers.l2(self.l2_reg)
            )
        else:
            self.projection = None
            
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        if self.projection is not None:
            inputs = self.projection(inputs)
            
        return layers.Add()([x, inputs])

class CancerDetector:
    def __init__(self, input_dim, num_classes, pathway_data=None):
        """
        Initialize the CancerDetector model
        
        Args:
            input_dim (int): Input dimension (number of features)
            num_classes (int): Number of cancer subtypes to predict
            pathway_data (dict, optional): Dictionary containing pathway information
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pathway_data = pathway_data
        self.scaler = StandardScaler()
        self.models = []
        self.model_weights = None
        self._build_ensemble()
        
    def _build_ensemble(self):
        """Build an ensemble of models with different architectures"""
        # Main deep learning model
        self.main_model = self._build_main_model()
        self.models.append(self.main_model)
        
        # Secondary model with different architecture
        self.secondary_model = self._build_secondary_model()
        self.models.append(self.secondary_model)
        
        # Random Forest model for ensemble
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def _build_main_model(self):
        """Build the main deep learning model"""
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Initial feature processing with L2 regularization
        x = layers.Dense(
            1024,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # First branch - Residual blocks with attention
        branch1 = ResidualBlock(512, dropout_rate=0.3)(x)
        branch1 = MultiHeadAttention(num_heads=8, key_dim=64)(branch1)
        branch1 = ResidualBlock(256, dropout_rate=0.3)(branch1)
        
        # Second branch - Deep processing with attention
        branch2 = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(x)
        branch2 = layers.BatchNormalization()(branch2)
        branch2 = layers.Dropout(0.3)(branch2)
        branch2 = MultiHeadAttention(num_heads=8, key_dim=64)(branch2)
        branch2 = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(branch2)
        
        # Combine branches
        x = layers.Concatenate()([branch1, branch2])
        
        # Global attention
        x = MultiHeadAttention(num_heads=8, key_dim=64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Final processing
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer with label smoothing
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            kernel_regularizer=regularizers.l2(0.01)
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use legacy Adam optimizer for better performance on M1/M2 Macs
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def _build_secondary_model(self):
        """Build a secondary model with different architecture"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Feature extraction
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Convolutional-like processing
        x = layers.Reshape((512, 1))(x)
        x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer with label smoothing
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use legacy Adam optimizer for better performance on M1/M2 Macs
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def mixup(self, x, y, alpha=0.2):
        """Apply mixup augmentation"""
        # Convert inputs to numpy arrays if they're DataFrames
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(y, 'values'):
            y = y.values
        
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = tf.shape(x)[0]
        index = tf.random.shuffle(tf.range(batch_size))
        
        mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
        mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
        
        return mixed_x.numpy(), mixed_y.numpy()
    
    def augment_data(self, X, y, noise_factor=0.05):
        """Apply advanced data augmentation techniques"""
        # Convert inputs to numpy arrays if they're DataFrames
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        X_aug = X.copy()
        y_aug = y.copy()
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_factor, X.shape)
        X_aug = X_aug + noise
        
        # Add random masking
        mask = np.random.random(X.shape) > 0.9
        X_aug[mask] = 0
        
        # Add random scaling
        scale = np.random.uniform(0.9, 1.1, X.shape)
        X_aug = X_aug * scale
        
        # Apply mixup
        X_mixup, y_mixup = self.mixup(X_aug, y_aug)
        
        # Combine original and augmented data
        X_combined = np.vstack([X, X_aug, X_mixup])
        y_combined = np.concatenate([y, y_aug, y_mixup])
        
        return X_combined, y_combined
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, callbacks=None):
        """Train the ensemble of models"""
        # Data augmentation
        X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
        
        # Train main model
        history_main = self.main_model.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks if callbacks else [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        )
        
        # Train secondary model
        history_secondary = self.secondary_model.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks if callbacks else [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        )
        
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        
        # Calculate model weights based on validation performance
        main_val_acc = max(history_main.history['val_accuracy'])
        secondary_val_acc = max(history_secondary.history['val_accuracy'])
        rf_val_acc = self.rf_model.score(X_val, y_val)
        
        total_acc = main_val_acc + secondary_val_acc + rf_val_acc
        self.model_weights = {
            'main': main_val_acc / total_acc,
            'secondary': secondary_val_acc / total_acc,
            'rf': rf_val_acc / total_acc
        }
        
        return history_main, history_secondary
    
    def predict(self, X):
        """Make ensemble predictions with dynamic weighting"""
        # Convert input to numpy array if it's a DataFrame
        if hasattr(X, 'values'):
            X = X.values
        
        # Get predictions from each model
        main_pred = self.main_model.predict(X)
        secondary_pred = self.secondary_model.predict(X)
        rf_pred = self.rf_model.predict_proba(X)
        
        # Convert predictions to numpy arrays and ensure correct shape
        main_pred = np.array(main_pred)
        secondary_pred = np.array(secondary_pred)
        rf_pred = np.array(rf_pred)
        
        # Get the target shape from the model with the most classes
        target_shape = (X.shape[0], self.num_classes)
        
        # Reshape predictions to match target shape
        if main_pred.shape != target_shape:
            if len(main_pred.shape) == 1:
                main_pred = np.expand_dims(main_pred, axis=1)
            if main_pred.shape[1] != target_shape[1]:
                temp = np.zeros(target_shape)
                temp[:, :main_pred.shape[1]] = main_pred
                main_pred = temp
        
        if secondary_pred.shape != target_shape:
            if len(secondary_pred.shape) == 1:
                secondary_pred = np.expand_dims(secondary_pred, axis=1)
            if secondary_pred.shape[1] != target_shape[1]:
                temp = np.zeros(target_shape)
                temp[:, :secondary_pred.shape[1]] = secondary_pred
                secondary_pred = temp
        
        # Handle Random Forest predictions shape
        if len(rf_pred.shape) == 3:  # If RF predictions are 3D
            rf_pred = np.mean(rf_pred, axis=0)  # Average across the extra dimension
        if rf_pred.shape != target_shape:
            if len(rf_pred.shape) == 1:
                rf_pred = np.expand_dims(rf_pred, axis=1)
            if rf_pred.shape[1] != target_shape[1]:
                temp = np.zeros(target_shape)
                temp[:, :rf_pred.shape[1]] = rf_pred
                rf_pred = temp
        
        # Use dynamic weights if available, otherwise use default weights
        if self.model_weights is not None:
            weights = self.model_weights
        else:
            weights = {'main': 0.4, 'secondary': 0.4, 'rf': 0.2}
        
        # Combine predictions with dynamic weights
        ensemble_pred = (
            weights['main'] * main_pred +
            weights['secondary'] * secondary_pred +
            weights['rf'] * rf_pred
        )
        
        # Normalize predictions to ensure they sum to 1
        ensemble_pred = ensemble_pred / np.sum(ensemble_pred, axis=1, keepdims=True)
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Convert y_test to numpy array if it's not already
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        
        # Ensure predictions are in the correct shape
        if len(y_pred.shape) == 3:
            y_pred = np.mean(y_pred, axis=0)  # Average predictions if we have multiple models
        
        # Calculate metrics
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
        
        # Calculate AUC and other metrics
        auc = tf.keras.metrics.AUC()
        auc.update_state(y_test, y_pred)
        auc_value = auc.result().numpy()
        
        precision = tf.keras.metrics.Precision()
        precision.update_state(y_test, y_pred)
        precision_value = precision.result().numpy()
        
        recall = tf.keras.metrics.Recall()
        recall.update_state(y_test, y_pred)
        recall_value = recall.result().numpy()
        
        f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value + 1e-7)
        
        return {
            'accuracy': float(accuracy),
            'auc': float(auc_value),
            'precision': float(precision_value),
            'recall': float(recall_value),
            'f1_score': float(f1)
        }
    
    def get_feature_importance(self, X):
        """Get feature importance scores"""
        # Get feature importance from random forest
        rf_importance = self.rf_model.feature_importances_
        
        # Get feature importance from neural networks using gradient-based approach
        main_importance = self._get_gradient_importance(self.main_model, X)
        secondary_importance = self._get_gradient_importance(self.secondary_model, X)
        
        # Combine importance scores with weights
        if self.model_weights is not None:
            weights = self.model_weights
        else:
            weights = {'main': 0.4, 'secondary': 0.4, 'rf': 0.2}
        
        combined_importance = (
            weights['main'] * main_importance +
            weights['secondary'] * secondary_importance +
            weights['rf'] * rf_importance
        )
        
        return combined_importance

    def _get_gradient_importance(self, model, X):
        """Calculate feature importance using gradient-based approach"""
        # Convert input to tensor
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor)
        
        # Calculate gradients
        gradients = tape.gradient(predictions, X_tensor)
        importance = tf.reduce_mean(tf.abs(gradients), axis=0)
        
        return importance.numpy() 