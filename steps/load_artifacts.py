from zenml.client import Client
from torch.utils.data import DataLoader
from zenml import step
from typing import Tuple

def load_artifact_from_pipeline(pipeline_name: str, step_name: str, output_name: str):
    client = Client()
    pipeline_obj = client.get_pipeline(pipeline_name)
    last_run = pipeline_obj.last_successful_run or pipeline_obj.last_run

    if not last_run:
        raise RuntimeError(f"No pipeline runs found for pipeline: {pipeline_name}")

    if step_name not in last_run.steps:
        raise RuntimeError(f"Step '{step_name}' not found in last pipeline run")

    step_obj = last_run.steps[step_name]
    outputs = step_obj.outputs

    if output_name not in outputs:
        raise RuntimeError(f"Output '{output_name}' not found in step '{step_name}' outputs")

    artifact_versions = outputs[output_name]

    if not artifact_versions:
        raise RuntimeError(f"No artifact versions found for output '{output_name}' in step '{step_name}'.")

    if isinstance(artifact_versions, list):
        if not artifact_versions:
            raise RuntimeError(f"No artifacts found for output '{output_name}'")
        artifact_version = artifact_versions[0]
    else:
        artifact_version = artifact_versions

    artifact_id = getattr(artifact_version, "id", None)
    if artifact_id is None:
        raise RuntimeError(f"Artifact version does not contain a valid ID.")

    try:
        artifact = client.get_artifact_version(artifact_id)
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve artifact version '{artifact_id}': {e}")

    try:
        loaded_data = artifact.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load artifact content from '{artifact.name}': {e}")

    return loaded_data


@step(enable_cache=False)
def load_training_pipeline() -> DataLoader:
    return load_artifact_from_pipeline("processing", "load", "training_batch")

@step(enable_cache=False)
def load_testing_pipeline() -> DataLoader:
    return load_artifact_from_pipeline("processing", "load", "testing_batch")

@step(enable_cache=False)
def load_trained_model():
    return load_artifact_from_pipeline("model_training", "training_model", "output")

@step(enable_cache=False)
def load_scores() -> Tuple[float, float, float, float]:
    accuracy = load_artifact_from_pipeline("model_evaluation_pipeline", "evaluation_model", "accuracy")
    precision = load_artifact_from_pipeline("model_evaluation_pipeline", "evaluation_model", "precision")
    recall = load_artifact_from_pipeline("model_evaluation_pipeline", "evaluation_model", "recall")
    f1_score = load_artifact_from_pipeline("model_evaluation_pipeline", "evaluation_model", "f1_score")

    return accuracy, precision, recall, f1_score
