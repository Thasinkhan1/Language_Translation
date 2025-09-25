import torch
from torch.utils.data import DataLoader, Subset
import random
from project_root.data import dataset
from project_root.data.data_loader import get_collate_fn
from project_root.config import config
from tokenizers import Tokenizer
import numpy as np

src_token = Tokenizer.from_file(config.src_tokenizer)
trg_token = Tokenizer.from_file(config.trg_tokenizer)
collate_fn = get_collate_fn(src_token, trg_token, pad_token=config.EXTRA_TOKEN_LIST[0])


def get_subset(dataset, n_samples,seed=42):
    np.random.seed(seed)
    subset_indices = np.random.choice(len(dataset), n_samples, replace=False)
    return Subset(dataset, subset_indices)

def train_loader():
    train_dataset = get_subset(dataset.load_train_data(), n_samples=100000)
    return DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

def val_loader():
    val_dataset = dataset.load_validation_data()
    return DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

def test_loader():
    test_dataset = dataset.load_test_data()
    return DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)