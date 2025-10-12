from pipeline import data_pipeline
from zenml import pipeline
from steps import training, evaluation
import logging

@pipeline(enable_cache=False)
def end_to_end_pipeline(epoch : int, learning_rate : float, data_path : str, num_of_labels, batch_size):
    try:
        training_data, testing_data = data_pipeline.processing(data_path, batch_size)
        logging.info("Successfully loaded and processed the training and testing data.")
    except Exception as e:
        logging.error(f"Error in data processing: {e}")
        raise
    
    try:
        trained_model = training.training_model(training_data, epoch, learning_rate, num_of_labels)
        logging.info("Model training complete.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise
    
    try:
        accuracy, precision, recall, score = evaluation.evaluation_model(trained_model, testing_data)
        logging.info(f"Evaluation complete. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {score}")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise
