import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, config: Dict):
        """Initialize the model trainer.
        
        Args:
            config (Dict): Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.callbacks = self._setup_callbacks()
        
    def _setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Set up training callbacks.
        
        Returns:
            List[tf.keras.callbacks.Callback]: List of callbacks
        """
        try:
            callbacks = []
            
            # Model checkpointing
            checkpoint_path = os.path.join(
                self.config['model_save_dir'],
                f'model_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
            )
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
            )
            
            # Early stopping
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['early_stopping_patience'],
                    restore_best_weights=True,
                    verbose=1
                )
            )
            
            # Learning rate reduction
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config['reduce_lr_factor'],
                    patience=self.config['reduce_lr_patience'],
                    min_lr=1e-6,
                    verbose=1
                )
            )
            
            # TensorBoard logging
            log_dir = os.path.join(
                self.config['log_dir'],
                f'tensorboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            callbacks.append(
                TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    update_freq='epoch'
                )
            )
            
            # CSV logging
            csv_path = os.path.join(
                self.config['log_dir'],
                f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
            callbacks.append(
                CSVLogger(
                    filename=csv_path,
                    separator=',',
                    append=False
                )
            )
            
            return callbacks
            
        except Exception as e:
            self.logger.error(f"Error setting up callbacks: {str(e)}")
            raise
            
    def _create_data_generator(self, X: np.ndarray, y: np.ndarray) -> tf.keras.preprocessing.image.ImageDataGenerator:
        """Create data generator for training.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            
        Returns:
            tf.keras.preprocessing.image.ImageDataGenerator: Data generator
        """
        try:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
            
            return datagen
            
        except Exception as e:
            self.logger.error(f"Error creating data generator: {str(e)}")
            raise
            
    def _create_learning_rate_schedule(self) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """Create learning rate schedule.
        
        Returns:
            tf.keras.optimizers.schedules.LearningRateSchedule: Learning rate schedule
        """
        try:
            initial_learning_rate = self.config['learning_rate']
            decay_steps = self.config['decay_steps']
            decay_rate = self.config['decay_rate']
            
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True
            )
            
            return lr_schedule
            
        except Exception as e:
            self.logger.error(f"Error creating learning rate schedule: {str(e)}")
            raise
            
    def train(self, model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        """Train the model.
        
        Args:
            model (tf.keras.Model): Model to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (Optional[np.ndarray]): Validation features
            y_val (Optional[np.ndarray]): Validation labels
            
        Returns:
            Dict: Training history
        """
        try:
            # Create data generator
            datagen = self._create_data_generator(X_train, y_train)
            
            # Create learning rate schedule
            lr_schedule = self._create_learning_rate_schedule()
            
            # Update model optimizer with learning rate schedule
            model.optimizer.learning_rate = lr_schedule
            
            # Train model
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=self.config['batch_size']),
                epochs=self.config['epochs'],
                validation_data=(X_val, y_val) if X_val is not None else None,
                callbacks=self.callbacks,
                verbose=1
            )
            
            self.logger.info("Model training completed successfully")
            return history.history
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
            
    def train_with_cross_validation(self, model: tf.keras.Model, X: np.ndarray, y: np.ndarray,
                                  n_splits: int = 5) -> List[Dict]:
        """Train model with k-fold cross validation.
        
        Args:
            model (tf.keras.Model): Model to train
            X (np.ndarray): Features
            y (np.ndarray): Labels
            n_splits (int): Number of folds
            
        Returns:
            List[Dict]: List of training histories
        """
        try:
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            histories = []
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                self.logger.info(f"Training fold {fold + 1}/{n_splits}")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                history = self.train(model, X_train, y_train, X_val, y_val)
                histories.append(history)
                
            return histories
            
        except Exception as e:
            self.logger.error(f"Error in cross validation training: {str(e)}")
            raise
            
    def evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the trained model.
        
        Args:
            model (tf.keras.Model): Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict: Evaluation metrics
        """
        try:
            metrics = model.evaluate(X_test, y_test, verbose=1)
            return dict(zip(model.metrics_names, metrics))
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def predict(self, model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            model (tf.keras.Model): Trained model
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            return model.predict(X, verbose=1)
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise 