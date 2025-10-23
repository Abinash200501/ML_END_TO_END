import logging
from zenml import step
import pandas as pd

class Ingest:

    def __init__(self, data_path):
        self.path = data_path

    def get_run(self):
        logging.info(f"Ingesting data from data path {self.path}")
        return pd.read_csv(self.path, sep='\t', names=['labels', 'Messages'])
    
@step(enable_cache=False)
def ingester(data_path : str) -> pd.DataFrame:

    try:
        ingest_data = Ingest(data_path)
        df = ingest_data.get_run()
        return df
    
    except Exception as e:
        logging.error(f"Error while ingesting data {e}")
        raise e


