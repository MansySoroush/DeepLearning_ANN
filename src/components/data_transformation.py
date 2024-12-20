import sys
import numpy as np 
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.configs.configurations import DataTransformationConfig
from src.utils import save_object

from src.components.data_ingestion import DataIngestion


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_objects(self):
        '''
        This function is responsible for data transformation      
        '''
        try:
            # Initialize the label encoder for Gender column
            label_encoder_gender = LabelEncoder()

            # Initialize the One-hot encoder for Geography column
            one_hot_encoder_geo = OneHotEncoder()

            return label_encoder_gender, one_hot_encoder_geo
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, cleaned_data_path):
        try:
            df = pd.read_csv(cleaned_data_path)

            logging.info("Read cleaned data completed")
            logging.info("Obtaining preprocessing object")

            label_encoder_gender, one_hot_encoder_geo = self.get_data_transformer_objects()         
            df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])

            geo_encoder = one_hot_encoder_geo.fit_transform(df[['Geography']]).toarray()
            one_hot_geo_columns = one_hot_encoder_geo.get_feature_names_out(['Geography'])
            geo_encoded_df = pd.DataFrame(geo_encoder, columns=one_hot_geo_columns)

            # Combine one hot encoder columns with the original data
            df = pd.concat([df.drop('Geography',axis=1), geo_encoded_df], axis=1)

            # Save the encoders
            save_object(
                file_path=self.data_transformation_config.label_encoder_gender_path,
                obj=label_encoder_gender
            )
            save_object(
                file_path=self.data_transformation_config.one_hot_encoder_geo_path,
                obj=one_hot_encoder_geo
            )

            logging.info("Complete preprocessing")

            return (
                self.data_transformation_config.label_encoder_gender_path,
                self.data_transformation_config.one_hot_encoder_geo_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    cleaned_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    label_encoder_gender, one_hot_encoder_geo = data_transformation.initiate_data_transformation(cleaned_data)
