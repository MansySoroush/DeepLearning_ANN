import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_regressor_models, split_train_test_sets
from src.configs.configurations import RegressorModelTrainerConfig

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from sklearn.metrics import mean_squared_error

class RegressorModelTrainer:
    def __init__(self):
        self.model_trainer_config = RegressorModelTrainerConfig()


    def initiate_model_trainer(self, cleaned_preprocessed_data, label_encoder_gender, one_hot_encoder_geo):
        try:
            target_column_name = "EstimatedSalary"

            logging.info("Start to do train test split process.")
            X_train, X_test, y_train, y_test = split_train_test_sets(data_path=cleaned_preprocessed_data, 
                                                                    target_column_name=target_column_name, 
                                                                    scaler_file_path=self.model_trainer_config.scaler_file_path)

            logging.info("Hyperparameter Tuning to find the best trained keras model.")

            # Define the grid search parameters
            param_grid = {
                'neurons': [16, 32, 64, 128],
                'layers': [1, 2],
                'learning_rate': [0.01, 0.001],
                'epochs': [50, 100],
                'batch_size': [16, 32]
            }

            keras_model_report:dict = evaluate_regressor_models(X_train = X_train, y_train = y_train, params = param_grid)
            
            logging.info("Complete hyperparameter Tuning.")

            best_model = keras_model_report['best_model']

            if best_model == None:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"------------Results-----------")
            logging.info(f"Best Params: {keras_model_report['best_param']}")
            logging.info(f"Best Score: {keras_model_report['best_score']}")
            logging.info(f"------------------------------")

            # Save the best  Model
            save_object(self.model_trainer_config.trained_model_file_path, obj= best_model)

            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            logging.info(f"Final Mean Squared Error: {mse}")
            logging.info(f"Complete Regressor Model Training")

            return mse           
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj = DataIngestion()
    cleaned_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    cleaned_preprocessed_data, label_encoder_gender, one_hot_encoder_geo = data_transformation.initiate_data_transformation(cleaned_data)

    model_trainer = RegressorModelTrainer()
    mse = model_trainer.initiate_model_trainer(cleaned_preprocessed_data, label_encoder_gender, one_hot_encoder_geo)
    print(f"Mean Squared Error of the Trained Model is: {mse}")
    

