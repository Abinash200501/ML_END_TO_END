import logging
import pandas as pd
from zenml import step
from sklearn.model_selection import train_test_split
from typing import Annotated, Tuple
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from steps import bert_tokenizer
import mlflow
from sklearn.utils import resample
from collections import Counter
from datetime import datetime

@step(enable_cache=False)
def clean(data : pd.DataFrame) -> pd.DataFrame:

    df = data.copy()
    original_df_shape = df.shape
    if df.isnull().sum().sum() != 0:             
        logging.info("Dropping missing values")
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        logging.info("No Missing values found")

    if df.duplicated().sum() != 0:
        logging.info("Duplicated found and dropping it")
        df.drop_duplicates(keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        logging.info("No duplicated found")
    
    logging.info(f"Reduced from {original_df_shape} to {df.shape} after cleaning.")
    return df


@step
def transform(data : pd.DataFrame) -> pd.DataFrame:

    data = data.copy()
    if 'labels' not in data.columns:
        raise ValueError("Missing column labels in dataset")
    
    else:
        unique_labels = sorted(data["labels"].unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        logging.info("Encoding the labels....")
        data['labels'] = data['labels'].map(label_map)

    return data

def oversample_dataset(data : pd.DataFrame, labels_name : str = 'labels') -> pd.DataFrame:

    data = data.copy()

    # Upsample minority class
    majority_class = data[data[labels_name] == 0]
    minority_class = data[data[labels_name] == 1]

    minority_resamples = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)

    balanced_dataset = pd.concat([majority_class, minority_resamples], axis=0)
    balanced_dataset.reset_index(drop = True, inplace = True)
    
    logging.info(f"Original dataset size : {data.shape}")
    logging.info(f"Balanced dataset size : {balanced_dataset.shape}")

    return balanced_dataset

@step
def validation(data : pd.DataFrame) -> pd.DataFrame:

    new_data = data.copy()
    logging.info("Checking class imbalance")

    ham_count = (new_data['labels'] == 0).sum()
    spam_count = (new_data['labels'] == 1).sum()


    is_imbalanced = abs(ham_count - spam_count) >= 100

    if is_imbalanced:
        
        label_counts = Counter(new_data['labels'])
        logging.info(f"Label counts before resampling :  {label_counts}")

        logging.info("Difference in count %d", ham_count - spam_count)
        logging.warning("Data is imbalanced")
        new_data = oversample_dataset(new_data)

        label_counts = Counter(new_data['labels'])
        logging.info(f"Label counts after resampling :  {label_counts}")

        is_imbalanced = False

    else:
        logging.info("Data is balanced")

    return new_data  
    
@step
def split(data : pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, "training_data"],
                                      Annotated[pd.DataFrame, "testing_data"]]:
    logging.info("Preparing the dataset with train and test")

    training_data, testing_data = train_test_split(data, test_size=0.2, random_state=42)

    return training_data, testing_data 


@step
def load(training, testing, batch_size) -> Tuple[Annotated[DataLoader, "training_batch"],
                                      Annotated[DataLoader, "testing_batch"]]:
    
    logging.info("Applying padding to get same size for inputs")

    collator = DataCollatorWithPadding(tokenizer=bert_tokenizer.tokenizer)

    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(testing, batch_size=batch_size, shuffle=False, collate_fn=collator)

    return train_loader, test_loader
