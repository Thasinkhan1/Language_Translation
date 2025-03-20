import pandas as pd
from config import config
import os

def load_data():
    
    data = pd.read_csv(os.path.join(config.DATASET_SAVE,config.DATASET_PATH),sep="\t",header=None,
    names=config.COLUMN_NAMES)
    
    return data

if __name__ == "__main__":
    load_data()