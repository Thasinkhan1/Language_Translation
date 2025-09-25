from datasets import load_dataset
from data import data_loader
from Preprocessing import tokenizer
from config import config
import numpy as np
from torch.utils.data import DataLoader, Subset


def load_data():
    data = load_dataset(config.DATASET_PATH)
    return data
    
data = load_data()
    
def load_train_data():        
    train_dataset = data_loader.TranslationDataset(data['train'], tokenizer.src_tokenizer(), tokenizer.trg_tokenizer())
    return train_dataset


def load_validation_data():
    validation_data = data_loader.TranslationDataset(data['validation'], tokenizer.src_tokenizer(), tokenizer.trg_tokenizer())
    
    return validation_data

def load_test_data():
    
    test_dataset  = data_loader.TranslationDataset(data['test'], tokenizer.src_tokenizer, tokenizer.trg_tokenizer())
    return test_dataset  


def get_subset(dataset, n_samples,seed=42):
    np.random.seed(seed)
    subset_indices = np.random.choice(len(dataset), n_samples, replace=False)
    return Subset(dataset, subset_indices)

