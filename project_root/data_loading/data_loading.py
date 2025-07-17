import pandas as pd # type: ignore
from config import config
import os
# import sys
# sys.path.append(os.path.abspath("project_root/data_loading"))

def load_data():
    data = pd.read_csv(os.path.join(config.DATASET_SAVE,config.DATASET_PATH),sep="\t",header=None,
    names=config.COLUMN_NAMES)
    return data
