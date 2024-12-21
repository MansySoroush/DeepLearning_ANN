import os
from dataclasses import dataclass

ARTIFACT_FOLDER_PATH = "artifacts"
CLASSIFICATION_FOLDER_PATH = "artifacts/classification"
REGRESSION_FOLDER_PATH = "artifacts/regression"
TRAIN_DATASET_FILE_NAME = "train.csv"
TEST_DATASET_FILE_NAME = "test.csv"
CLEANED_DATASET_FILE_NAME = "cleaned_data.csv"
CLEANED_PREPROCESSED_DATASET_FILE_NAME = "cleaned_preprocessed_data.csv"
TRAINED_MODEL_FILE_NAME = "model.pkl"
TRAINED_MODEL_KERAS_FILE_NAME = "model.keras"
LABEL_ENCODER_GENDER_FILE_NAME = "label_encoder_gender.pkl"
ONE_HOT_ENCODER_GEO_FILE_NAME = "one_hot_encoder_geo.pkl"
SCALER_FILE_NAME = "scaler.pkl"

@dataclass
class DataIngestionConfig:
    cleaned_data_path: str = os.path.join(ARTIFACT_FOLDER_PATH, CLEANED_DATASET_FILE_NAME)


@dataclass
class DataTransformationConfig:
    label_encoder_gender_path: str = os.path.join(ARTIFACT_FOLDER_PATH, LABEL_ENCODER_GENDER_FILE_NAME)
    one_hot_encoder_geo_path: str = os.path.join(ARTIFACT_FOLDER_PATH, ONE_HOT_ENCODER_GEO_FILE_NAME)
    cleaned_preprocessed_data_path: str = os.path.join(ARTIFACT_FOLDER_PATH, CLEANED_PREPROCESSED_DATASET_FILE_NAME)

@dataclass
class ClassifierModelTrainerConfig:
    trained_model_file_path = os.path.join(CLASSIFICATION_FOLDER_PATH, TRAINED_MODEL_FILE_NAME)
    trained_model_keras_file_path = os.path.join(CLASSIFICATION_FOLDER_PATH, TRAINED_MODEL_KERAS_FILE_NAME)
    scaler_file_path = os.path.join(CLASSIFICATION_FOLDER_PATH, SCALER_FILE_NAME)

@dataclass
class RegressorModelTrainerConfig:
    trained_model_file_path = os.path.join(REGRESSION_FOLDER_PATH, TRAINED_MODEL_FILE_NAME)
    trained_model_keras_file_path = os.path.join(REGRESSION_FOLDER_PATH, TRAINED_MODEL_KERAS_FILE_NAME)
    scaler_file_path = os.path.join(REGRESSION_FOLDER_PATH, SCALER_FILE_NAME)

@dataclass
class ClassifierPredictPipelineConfig:
    trained_model_keras_file_path = os.path.join(CLASSIFICATION_FOLDER_PATH, TRAINED_MODEL_KERAS_FILE_NAME)
    label_encoder_gender_path: str = os.path.join(ARTIFACT_FOLDER_PATH, LABEL_ENCODER_GENDER_FILE_NAME)
    one_hot_encoder_geo_path: str = os.path.join(ARTIFACT_FOLDER_PATH, ONE_HOT_ENCODER_GEO_FILE_NAME)
    scaler_file_path = os.path.join(CLASSIFICATION_FOLDER_PATH, SCALER_FILE_NAME)

@dataclass
class RegressorPredictPipelineConfig:
    trained_model_keras_file_path = os.path.join(REGRESSION_FOLDER_PATH, TRAINED_MODEL_KERAS_FILE_NAME)
    label_encoder_gender_path: str = os.path.join(ARTIFACT_FOLDER_PATH, LABEL_ENCODER_GENDER_FILE_NAME)
    one_hot_encoder_geo_path: str = os.path.join(ARTIFACT_FOLDER_PATH, ONE_HOT_ENCODER_GEO_FILE_NAME)
    scaler_file_path = os.path.join(REGRESSION_FOLDER_PATH, SCALER_FILE_NAME)
