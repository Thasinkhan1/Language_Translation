from config import config
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from data import dataset
from Preprocessing import tokenizer
# import sys
# sys.path.append(os.path.abspath("project_root/data_loading"))

class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, trg_tokenizer, max_len=50):
        self.data = data
        self.src_tok = src_tokenizer
        self.trg_tok = trg_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = int(idx)
        src = self.data[idx]['translation']['en']
        trg = self.data[idx]['translation']['hi']

        # Encode
        src_ids = [self.src_tok.token_to_id("<SOS>")] + self.src_tok.encode(src).ids[:self.max_len-2] + [self.src_tok.token_to_id("<EOS>")]
        trg_ids = [self.trg_tok.token_to_id("<SOS>")] + self.trg_tok.encode(trg).ids[:self.max_len-2] + [self.trg_tok.token_to_id("<EOS>")]

        return torch.tensor(src_ids), torch.tensor(trg_ids)
    
from torch.nn.utils.rnn import pad_sequence

src_token = tokenizer.src_tokenizer()
trg_token = tokenizer.trg_tokenizer()

pad_idx = src_token.token_to_id(config.EXTRA_TOKEN_LIST[0])

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=src_token.token_to_id(config.EXTRA_TOKEN_LIST[0]), batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=trg_token.token_to_id(config.EXTRA_TOKEN_LIST[0]), batch_first=True)
    
    return src_batch, trg_batch

train_dataset = dataset.load_train_data()
val_dataset = dataset.load_validation_data()
test_dataset = dataset.load_test_data()

train_dataset_small = dataset.get_subset(train_dataset, n_samples=100000)  # 100k pairs
#val_dataset_small = get_subset(val_dataset, n_samples=5000)        # 5k pairs

def train_loader():
    train_load = DataLoader(train_dataset_small, batch_size=64, shuffle=True, collate_fn=collate_fn)
    return train_load

def val_loader():
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,collate_fn=collate_fn)
    return val_loader


def test_loader():
    test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn)
    
# print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")