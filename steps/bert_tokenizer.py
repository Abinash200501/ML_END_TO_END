from transformers import AutoTokenizer
from datasets import Dataset
from zenml import step
from typing import Annotated, Tuple
import logging

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_data(data):
    encoding = tokenizer(data['Messages'], truncation=True, padding=True, max_length=512)
    return encoding


@step
def tokenized_with_step(train_data, test_data) -> Tuple[Annotated[Dataset, "training_dataset"],
                                       Annotated[Dataset, "test_dataset"]]:
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    for dataset, name in [(train_dataset, "training"), (test_dataset, "testing")]:
        if "Messages" not in dataset.column_names:
            raise ValueError(f"Message column missing in {name} dataset")
        if "labels" not in dataset.column_names:
            raise ValueError(f"Labels column missing in {name} dataset")
        
        logging.info(f"{name} datasets are verfied both Messages and labels columns are present")


    tokenized_train = train_dataset.map(tokenize_data, batched=True, remove_columns=["Messages", '__index_level_0__'])
    tokenized_test = test_dataset.map(tokenize_data, batched=True, remove_columns=["Messages", '__index_level_0__'])

    tokenized_train.set_format("torch", columns=["input_ids", "token_type_ids","attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "token_type_ids","attention_mask", "labels"])
    
    return tokenized_train, tokenized_test

