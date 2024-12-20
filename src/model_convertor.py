from src.utils import load_object, save_model
from src.configs.configurations import ClassifierModelTrainerConfig, RegressorModelTrainerConfig


if __name__=="__main__":
    model_trainer_config = ClassifierModelTrainerConfig()
    classifier_keras_model = load_object(file_path=model_trainer_config.trained_model_file_path)
    save_model(classifier_keras_model, model_trainer_config.trained_model_keras_file_path)

    model_trainer_config = RegressorModelTrainerConfig()
    regressor_keras_model = load_object(file_path=model_trainer_config.trained_model_file_path)
    save_model(regressor_keras_model, model_trainer_config.trained_model_keras_file_path)


