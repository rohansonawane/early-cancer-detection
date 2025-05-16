import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input,
    Conv1D, MaxPooling1D, Flatten, Concatenate,
    LSTM, Bidirectional, Attention, MultiHeadAttention,
    LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from typing import Dict, List, Tuple, Optional
import logging

class ResidualBlock(tf.keras.layers.Layer):
    """Residual block for deep neural networks."""
    
    def __init__(self, units: int, dropout_rate: float = 0.3):
        super(ResidualBlock, self).__init__()
        self.dense1 = Dense(units, activation='relu')
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(dropout_rate)
        self.dense2 = Dense(units, activation='relu')
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(dropout_rate)
        self.add = Add()
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.add([x, inputs])

class AttentionBlock(tf.keras.layers.Layer):
    """Attention block for focusing on important features."""
    
    def __init__(self, units: int):
        super(AttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=4, key_dim=units)
        self.norm = LayerNormalization()
        
    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        return self.norm(inputs + attention_output)

class CancerDetectionModel:
    def __init__(self, input_dim: int, num_classes: int, config: Dict):
        """Initialize the cancer detection model.
        
        Args:
            input_dim (int): Input dimension
            num_classes (int): Number of output classes
            config (Dict): Model configuration
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build the model architecture.
        
        Returns:
            Model: Compiled Keras model
        """
        try:
            # Input layer
            inputs = Input(shape=(self.input_dim,))
            
            # Initial dense layer
            x = Dense(
                self.config['hidden_layers'][0],
                activation='relu',
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            )(inputs)
            x = BatchNormalization()(x)
            x = Dropout(self.config['dropout_rate'])(x)
            
            # Residual blocks
            for units in self.config['hidden_layers'][1:-1]:
                x = ResidualBlock(units, self.config['dropout_rate'])(x)
            
            # Attention mechanism
            x = AttentionBlock(self.config['hidden_layers'][-1])(x)
            
            # Output layer
            outputs = Dense(
                self.num_classes,
                activation='softmax',
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            )(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.config['learning_rate']),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            raise
            
    def _build_convolutional_model(self) -> Model:
        """Build a convolutional model architecture.
        
        Returns:
            Model: Compiled Keras model
        """
        try:
            # Input layer
            inputs = Input(shape=(self.input_dim, 1))
            
            # Convolutional layers
            x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            
            x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            
            x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            
            # Flatten and dense layers
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            
            # Output layer
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.config['learning_rate']),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error building convolutional model: {str(e)}")
            raise
            
    def _build_lstm_model(self) -> Model:
        """Build an LSTM model architecture.
        
        Returns:
            Model: Compiled Keras model
        """
        try:
            # Input layer
            inputs = Input(shape=(self.input_dim, 1))
            
            # LSTM layers
            x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = Bidirectional(LSTM(64))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Dense layers
            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Output layer
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.config['learning_rate']),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error building LSTM model: {str(e)}")
            raise
            
    def get_model_summary(self) -> str:
        """Get model summary as a string.
        
        Returns:
            str: Model summary
        """
        try:
            string_list = []
            self.model.summary(print_fn=lambda x: string_list.append(x))
            return '\n'.join(string_list)
        except Exception as e:
            self.logger.error(f"Error getting model summary: {str(e)}")
            raise
            
    def save_model(self, filepath: str):
        """Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            self.model.save(filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, filepath: str):
        """Load a model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise 