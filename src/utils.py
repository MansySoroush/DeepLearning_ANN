import os
import sys
import dill
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from tensorflow.keras.callbacks import EarlyStopping
from src.custom_keras_classifier import CustomKerasClassifier
from src.custom_keras_regressor import CustomKerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import tensorflow as tf

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

def evaluate_classifier_models(X_train, y_train, params):
    try:
        report = {}

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model = CustomKerasClassifier(callbacks=[early_stopping_callback])

        grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=3, verbose=1)
        grid_result = grid.fit(X_train, y_train)
        best_model = grid_result.best_estimator_

        report = {
            "best_param": grid_result.best_params_,
            "best_score": grid_result.best_score_,
            "best_model": best_model
        }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_regressor_models(X_train, y_train, params):
    try:
        report = {}

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model = CustomKerasRegressor(callbacks=[early_stopping_callback])

        # Custom scorer for GridSearchCV
        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

        # Use the custom scorer in GridSearchCV
        grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=3, verbose=1, scoring=mse_scorer)
        grid_result = grid.fit(X_train, y_train)
        best_model = grid_result.best_estimator_

        report = {
            "best_param": grid_result.best_params_,
            "best_score": grid_result.best_score_,
            "best_model": best_model
        }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def split_train_test_sets(data_path, target_column_name, scaler_file_path):
    try:
        df = pd.read_csv(data_path)

        logging.info("Read cleaned preprocessed data completed")

        # Dependent feature
        y = df[target_column_name]

        # Independent feature
        X = df.drop(columns=[target_column_name],axis=1)
        
        logging.info("Train test split initiated")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logging.info(f"Training data size: {len(X_train)}")
        logging.info(f"Test data size: {len(X_test)}")

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save the scaler
        save_object(file_path=scaler_file_path, obj=scaler)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise CustomException(e, sys)
    
def save_model(keras_model, file_path):
    try:
        if keras_model.model_:
            keras_model.model_.save(file_path)
            print(f"Model saved to {file_path}")
        else:
            raise CustomException("No model found. Train the model before saving.")
        
    except Exception as e:
        raise CustomException(e, sys)

def get_processed_data_frame_to_predict(model, label_encoder_gender, one_hot_encoder_geo, scaler, features_df):
    try:    
        # One-hot encode 'Geography'
        geo_encoded = one_hot_encoder_geo.transform([features_df['Geography']]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

        # label encode 'Gender'
        features_df['Gender'] = label_encoder_gender.transform(features_df['Gender'])

        # concatination with one hot encoded 
        features_df=pd.concat([features_df.drop("Geography",axis=1), geo_encoded_df],axis=1)

        # Scaling the input data
        features_df = scaler.transform(features_df)
        
        return features_df
    
    except Exception as e:
        raise CustomException(e, sys)

def get_objects_to_predict(trained_model_keras_file_path, label_encoder_gender_path, 
                            one_hot_encoder_geo_path, scaler_file_path):
    try:
        logging.info("Loading keras model...")
        model = tf.keras.models.load_model(trained_model_keras_file_path)

        logging.info("Loading label encoder gender...")
        label_encoder_gender = load_object(file_path=label_encoder_gender_path)

        logging.info("Loading one hot encoder geography...")
        one_hot_encoder_geo = load_object(file_path=one_hot_encoder_geo_path)

        logging.info("Loading scaler...")
        scaler = load_object(file_path=scaler_file_path)
            
        return model, label_encoder_gender, one_hot_encoder_geo, scaler
    
    except Exception as e:
        raise CustomException(e, sys)
