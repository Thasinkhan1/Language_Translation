# project_root/data/data_loader.py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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

        src_ids = [self.src_tok.token_to_id("<SOS>")] + self.src_tok.encode(src).ids[:self.max_len-2] + [self.src_tok.token_to_id("<EOS>")]
        trg_ids = [self.trg_tok.token_to_id("<SOS>")] + self.trg_tok.encode(trg).ids[:self.max_len-2] + [self.trg_tok.token_to_id("<EOS>")]

        return torch.tensor(src_ids), torch.tensor(trg_ids)


def get_collate_fn(src_tokenizer, trg_tokenizer, pad_token="<PAD>"):
    pad_idx_src = src_tokenizer.token_to_id(pad_token)
    pad_idx_trg = trg_tokenizer.token_to_id(pad_token)

    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, padding_value=pad_idx_src, batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=pad_idx_trg, batch_first=True)
        return src_batch, trg_batch

    return collate_fn
