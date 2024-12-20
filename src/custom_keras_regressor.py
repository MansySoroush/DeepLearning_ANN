import sys
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from src.exception import CustomException
from sklearn.metrics import mean_squared_error


class CustomKerasRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, neurons=32, layers=1, learning_rate=0.01, epochs=50, batch_size=32, verbose=1, callbacks=None, validation_split = 0.2):
        self.neurons = neurons
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.callbacks = callbacks
        self._estimator_type = "regressor"
        self.validation_split = validation_split

    def build_model(self):
        try:
            model = Sequential()
            model.add(Input(shape=(self.input_shape_,)))  # Explicitly define the input shape
            model.add(Dense(self.neurons, activation='relu'))
            
            for _ in range(self.layers - 1):
                model.add(Dense(self.neurons, activation='relu'))
            
            model.add(Dense(1, activation='linear'))  # Output layer for regression
            
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def fit(self, X, y):
        try:
            self.input_shape_ = X.shape[1]
            self.model_ = self.build_model()
            self.model_.fit(
                X, 
                y, 
                epochs=self.epochs, 
                batch_size=self.batch_size, 
                verbose=self.verbose, 
                callbacks=self.callbacks,
                validation_split = self.validation_split
            )
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X):
        try:
            predictions = self.model_.predict(X)
            return np.squeeze(predictions)
        except Exception as e:
            raise CustomException(e, sys)

    def score(self, X, y):
        try:
            predictions = self.predict(X)
            mse = mean_squared_error(y, predictions)
            return -mse  # Negative MSE because GridSearchCV maximizes the score
        except Exception as e:
            raise CustomException(e, sys)
