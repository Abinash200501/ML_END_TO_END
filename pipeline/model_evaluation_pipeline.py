from steps.load_artifacts import load_testing_pipeline, load_trained_model
from steps.evaluation import evaluation_model
from zenml import pipeline

@pipeline
def model_evaluation_pipeline():
    testing_batch = load_testing_pipeline()
    model = load_trained_model()
    accuracy, precision, recall, f1_score = evaluation_model(model, testing_batch)