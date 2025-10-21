import logging
from zenml import step
from typing import Tuple, Annotated
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from strategy import metrics
import mlflow
from tqdm.auto import tqdm
import pickle
from pathlib import Path
from steps import bert_tokenizer
import pandas as pd

@step(enable_cache=False)
def evaluation_model(model : BertForSequenceClassification, 
                     testing_batch : DataLoader) -> Tuple[Annotated[float, "accuracy"],
                                                               Annotated[float, "precision"],
                                                               Annotated[float, "recall"],
                                                               Annotated[float, "f1_score"]]:

    logging.info("Evaluation phase started")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    y_true = []
    preds = []

    misclassified = []

    no_of_steps = len(testing_batch)

    for batch in tqdm(testing_batch, total=no_of_steps):
        batch = {k:v.to(device) for k, v in batch.items()}

        labels = batch['labels'].float()

        with torch.no_grad():
            output = model(**batch)
        
        logits = output.logits
        probs = torch.sigmoid(logits).squeeze(-1)
        predictions= (probs >= 0.5).long()
        preds.extend(predictions.cpu().tolist())          
        y_true.extend(batch['labels'].cpu().tolist())     
        
        for true_label, predicted_label in zip(labels.tolist(), predictions.tolist()):
            if true_label != predicted_label:
                misclassified.append({"true_label": true_label,
                                      "predicted_label": predicted_label})
                
    if misclassified:
        df_misclassified = pd.DataFrame(misclassified)
        df_misclassified.to_csv("misclassified_samples_test.csv", index=False)
        logging.info(f"Saved {len(df_misclassified)} misclassified samples to CSV.")
    else:
        logging.info("No misclassified samples found")

    logging.info("Testing is finished")

    TP = TN = FP = FN = 0

    for i in range(len(preds)):
        if y_true[i] == 1 and preds[i] == 1:
            TP += 1
        elif y_true[i] == 0 and preds[i] == 0:
            TN += 1
        elif y_true[i] == 1 and preds[i] == 0:
            FN += 1
        elif y_true[i] == 0 and preds[i] == 1:
            FP += 1

    mlflow.log_metrics({"true_positives":TP,
                        "true_negatives":TN,
                        "false_positive":FP,
                        "false_negative":FN
    })

    accuracy, precision, recall, f1_score  = metrics.confusion_matrix(TP, TN, FP, FN)

    mlflow.log_metrics({"accuracy":accuracy, "precision":precision, 
                        "recall":recall, "f1_score":f1_score})

    run_id = mlflow.active_run().info.run_id
    run = mlflow.get_run(run_id=run_id)

    params = run.data.params

    if accuracy >= 0.90:
        model_dict = {"model":model, "metrics" : {"accuracy": accuracy, "f1_score": f1_score}, 
                      "params": params,
                      "mlflow_run_id": run_id,"tokenizer":bert_tokenizer.tokenizer}

        save_path = Path("saved_model")
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save(model_dict, save_path / "model.pkl")
        logging.info("Model is saved for deployment")
    else:
        logging.info("Not good enough to save the model")
        
    return accuracy, precision, recall, f1_score


