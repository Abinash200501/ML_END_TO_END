import logging
from zenml import step
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
import torch
import mlflow
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
from collections import Counter
import torch.nn as nn
from steps.bert_tokenizer import tokenizer

@step(enable_cache=True)
def training_model(training_data: DataLoader, epoch: int, lr: float,  num_of_labels) -> BertForSequenceClassification:

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_of_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fun = nn.BCEWithLogitsLoss()

    mlflow.log_param("loss_function","BCEWithLogitsLoss (unweighted)")

    no_of_steps = epoch * len(training_data)
    progress_bar = tqdm(range(no_of_steps))

    label_counter = Counter()

    for batch in training_data:
        labels = batch['labels']
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        label_counter.update(labels)

    logging.info(f"Label distribution before training: {label_counter}")

    logging.info("Model is training")
    model.train()


    for epoch_idx in range(epoch):
        total_loss = 0

        for i, batch in enumerate(training_data):

            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch['labels'].float()

            output = model(**batch)
            logits = output.logits.squeeze(1)

            loss = loss_fun(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            progress_bar.update(1)
            total_loss += loss.item()

            if i % 100 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                reserved = torch.cuda.memory_reserved() / 1024**2    # MB
                print(f"[GPU Memory] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
                print(f"Epoch {epoch_idx+1}/{epoch}, Step : {i}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(training_data)
        mlflow.log_metric("average_loss", avg_loss, step=epoch_idx)
        logging.info(f"Epoch {epoch_idx + 1} average loss: {avg_loss:.5f}")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in training_data:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= 0.5)
            all_preds.extend(preds.cpu().long().tolist())
            all_labels.extend(batch['labels'].cpu().long().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    mlflow.log_metric("train_accuracy", accuracy, step=epoch_idx)
    logging.info(f"Epoch {epoch_idx+1} Accuracy: {accuracy:.4f}")


    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Training is finished")
 
    print("All labels : ", all_labels[:10])
    print("Counter labels : ", Counter(all_labels))
    print("All predicted : ", all_preds[:10])
    print("Counter predicted labels : ", Counter(all_preds))

    components = {"model":model, "tokenizer":tokenizer}

    mlflow.transformers.log_model(transformers_model=components, task="text-classification", artifact_path="model")

    
    save_path = Path("saved_model")
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / "trained_model.pkl"

    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    mlflow.log_artifact(str(model_file))
    logging.info(f"Model saved to: {model_file.resolve()} and logged to MLflow.")
    
    return model


