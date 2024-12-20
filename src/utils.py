import os
import sys
import dill
import numpy as np

from src.exception import CustomException
from src.logger import logging

from tensorflow.keras.callbacks import EarlyStopping
from src.custom_keras_classifier import CustomKerasClassifier
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_classifier_models(X_train, y_train, X_test, y_test, params):
    try:
        report = {}

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model = CustomKerasClassifier(callbacks=[early_stopping_callback])

        grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=3, verbose=1)
        grid_result = grid.fit(X_train, y_train)

        # Print the best parameters and the best score
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        best_model = grid_result.best_estimator_

        report = {
            "best_param": grid_result.best_params_,
            "best_score": grid_result.best_score_,
            "best_model": best_model
        }

        return report

    except Exception as e:
        raise CustomException(e, sys)
