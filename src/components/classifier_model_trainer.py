import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_classifier_models
from src.configs.configurations import ClassifierModelTrainerConfig

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class ClassifierModelTrainer:
    def __init__(self):
        self.model_trainer_config = ClassifierModelTrainerConfig()


    def initiate_model_trainer(self, cleaned_preprocessed_data, label_encoder_gender, one_hot_encoder_geo):
        try:
            df = pd.read_csv(cleaned_preprocessed_data)

            logging.info("Read cleaned preprocessed data completed")

            target_column_name = "Exited"

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
            save_object(
                file_path=self.model_trainer_config.scaler_file_path,
                obj=scaler
            )

            logging.info("Hyperparameter Tuning to find the best trained keras model.")

            # Define the grid search parameters
            param_grid = {
                'neurons': [16, 32, 64, 128],
                'layers': [1, 2],
                'learning_rate': [0.01, 0.001],
                'epochs': [50, 100],
                'batch_size': [16, 32]
            }

            keras_model_report:dict = evaluate_classifier_models(X_train = X_train, y_train = y_train,
                                                X_test = X_test, y_test = y_test,
                                                params = param_grid)
            
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
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Final Accuracy: {accuracy}")
            logging.info(f"Complete Classifier Model Training")

            return accuracy           
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj = DataIngestion()
    cleaned_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    cleaned_preprocessed_data, label_encoder_gender, one_hot_encoder_geo = data_transformation.initiate_data_transformation(cleaned_data)

    model_trainer = ClassifierModelTrainer()
    accuracy = model_trainer.initiate_model_trainer(cleaned_preprocessed_data, label_encoder_gender, one_hot_encoder_geo)
    print(f"Accuracy Score of the Trained Model is: {accuracy}")
    

