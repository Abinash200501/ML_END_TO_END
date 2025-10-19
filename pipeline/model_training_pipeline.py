from steps.training import training_model
from steps.load_artifacts import load_training_pipeline, load_testing_pipeline
from steps.evaluation import evaluation_model
from zenml import pipeline

@pipeline
def model_training(no_of_epoch, lr, num_of_labels):
    training_batch = load_training_pipeline()
    model = training_model(training_batch, no_of_epoch, lr, num_of_labels)
    testing_batch = load_testing_pipeline()
    accuracy, precision, recall, score = evaluation_model(model, testing_batch)