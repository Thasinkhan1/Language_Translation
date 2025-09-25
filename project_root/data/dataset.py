from datasets import load_dataset
from project_root.config import config
from project_root.data import data_loader
from tokenizers import Tokenizer

def load_data():
    data = load_dataset(config.DATASET_PATH)
    return data
    
data = load_data()
src_token = Tokenizer.from_file(config.src_tokenizer)
trg_token = Tokenizer.from_file(config.trg_tokenizer)
    
def load_train_data():        
    train_dataset = data_loader.TranslationDataset(data['train'], src_token,trg_token)
    return train_dataset


def load_validation_data():
    validation_data = data_loader.TranslationDataset(data['validation'],src_token,trg_token)
    
    return validation_data

def load_test_data():
    test_dataset  = data_loader.TranslationDataset(data['test'],src_token, trg_token)
    return test_dataset  