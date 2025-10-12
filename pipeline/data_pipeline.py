from zenml import pipeline
from typing import Annotated, Tuple
from steps import data_load, ingest_data, bert_tokenizer
from torch.utils.data import DataLoader


@pipeline
def processing(data_path : str, batch_size : int) -> Tuple[Annotated[DataLoader, "training_batch"],
                                         Annotated[DataLoader, "testing_batch"]]:
    
    data = ingest_data.ingester(data_path)
    cleaned_data = data_load.clean(data)
    transformed_data = data_load.transform(cleaned_data)
    validated_data = data_load.validation(transformed_data)
    training_set, test_set= data_load.split(validated_data)
    tokeninzed_train, tokenized_test = bert_tokenizer.tokenized_with_step(training_set, test_set)
    training_batch, testing_batch = data_load.load(tokeninzed_train, tokenized_test, batch_size)

    return training_batch, testing_batch