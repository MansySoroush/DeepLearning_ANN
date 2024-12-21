import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import get_processed_data_frame_to_predict, get_objects_to_predict
from src.configs.configurations import ClassifierPredictPipelineConfig, RegressorPredictPipelineConfig

import streamlit as st

@st.cache_resource
def load_model_objects(predict_pipeline_config):
    try:
        model, label_encoder_gender, one_hot_encoder_geo, scaler = get_objects_to_predict(
            trained_model_keras_file_path=predict_pipeline_config.trained_model_keras_file_path,
            label_encoder_gender_path=predict_pipeline_config.label_encoder_gender_path,
            one_hot_encoder_geo_path=predict_pipeline_config.one_hot_encoder_geo_path,
            scaler_file_path=predict_pipeline_config.scaler_file_path
        )
        return model, label_encoder_gender, one_hot_encoder_geo, scaler
    except Exception as e:
        raise CustomException(e,sys)


class PredictPipeline:
    def __init__(self, is_classifier):
        self.is_classifier = is_classifier
        self.predict_pipeline_config = ClassifierPredictPipelineConfig() if is_classifier else RegressorPredictPipelineConfig()
        self.label_encoder_gender = None
        self.one_hot_encoder_geo = None
        self.scaler = None
        self.model = None

    def predict(self, features_df):
        try:
            self.model, self.label_encoder_gender, self.one_hot_encoder_geo, self.scaler = self.get_objects()

            if (self.model == None) or (self.label_encoder_gender == None) or (self.one_hot_encoder_geo == None) or (self.scaler == None):
                raise CustomException("Prediction Objects are None!!!!!")
            
            features_df = get_processed_data_frame_to_predict(model= self.model, 
                                                            label_encoder_gender= self.label_encoder_gender, 
                                                            one_hot_encoder_geo= self.one_hot_encoder_geo, 
                                                            scaler= self.scaler, 
                                                            features_df= features_df)
            logging.info("Ready to Predict.")

            y_pred = self.model.predict(features_df)

            logging.info(f"y_pred: {y_pred}")
            
            return y_pred
        
        except Exception as e:
            raise CustomException(e,sys)
            
    def get_objects(self):
        try:
            if (self.model == None) or (self.label_encoder_gender == None) or (self.one_hot_encoder_geo == None) or (self.scaler == None):
                self.model, self.label_encoder_gender, self.one_hot_encoder_geo, self.scaler = load_model_objects(self.predict_pipeline_config)
                
            return self.model, self.label_encoder_gender, self.one_hot_encoder_geo, self.scaler
        
        except Exception as e:
            raise CustomException(e,sys)
