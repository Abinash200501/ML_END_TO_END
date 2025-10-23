import click
import mlflow
from pipeline.data_pipeline import processing
from pipeline.model_training_pipeline import model_training
from pipeline.model_evaluation_pipeline import model_evaluation_pipeline
from pipeline.end_to_end import end_to_end_pipeline
from zenml.client import Client


def check_trained_model_exists():
    "Checks if a successful training pipeline run exists before evaluating."
    client = Client()
    training_pipeline = client.get_pipeline("model_training")
    last_run = training_pipeline.last_successful_run

    if not last_run:
        raise RuntimeError(
            "No successful model training run found.\n"
            "Please run with --train-model or --end-to-end before evaluating."
        )


@click.command(help="Run various stages of the ZenML pipeline.")
@click.option("--load-data", is_flag=True, default=False, help="Create the dataset.")
@click.option("--train-model", is_flag=True, default=False, help="Run the training pipeline.")
@click.option("--evaluate-model", is_flag=True, default=False, help="Evaluate the last trained model.")
@click.option("--end-to-end", is_flag=True, default=False, help="Run all pipelines in sequence.")
@click.option("--path", default=None, type=click.STRING, help="Path to the dataset.")
@click.option("--learning-rate", default=0.0002, type=click.FLOAT, help="Learning rate for training.")
@click.option("--num-epochs", default=2, type=click.INT, help="Number of training epochs.")
@click.option("--num-of-labels", default=1, type=click.INT, help="Number of classification labels.")
@click.option("--batch-size", default=16, type=click.INT, help="Batch size for data loading.")

def main(
    load_data: bool,
    path: str,
    train_model: bool,
    num_epochs: int,
    learning_rate: float,
    num_of_labels: int,
    evaluate_model: bool,
    end_to_end: bool,
    batch_size: int
):
    print(f"Flags - load_data={load_data}, train_model={train_model}, "
          f"evaluate_model={evaluate_model}, end_to_end={end_to_end}")

    if train_model:
        run_name = "training_pipeline"
    elif end_to_end:
        run_name = "end_to_end_pipeline"
    elif evaluate_model:
        run_name = "evaluation_pipeline"
    elif load_data:
        run_name = "data_pipeline"
    else:
        run_name = "manual_run"

    with mlflow.start_run(run_name=run_name):

        if train_model or end_to_end:
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)

        if load_data:
            processing(path, batch_size)

        if train_model:
            model_training(num_epochs, learning_rate, num_of_labels)

        if evaluate_model:
            check_trained_model_exists()
            model_evaluation_pipeline()

        if end_to_end:
            end_to_end_pipeline(num_epochs, learning_rate, path, num_of_labels, batch_size)


if __name__ == "__main__":

    mlflow.set_tracking_uri("http://ec2-3-89-33-158.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment("Spam_Classifier_Experiment")
    main()
