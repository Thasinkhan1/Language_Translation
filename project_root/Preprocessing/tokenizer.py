from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from data import dataset
from config import config



def train_tokenizer(sentences, vocab_size=30000, lang=config.SRC_LANG):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=config.EXTRA_TOKEN_LIST
    )
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(f"{lang}_tokenizer.json")
    return tokenizer


def src_tokenizer():
    data = dataset.load_data()
    english_sentences = [x['translation'][config.SRC_LANG] for x in data['train']]
    src_token = train_tokenizer(english_sentences, lang=config.SRC_LANG)
    return src_token

def trg_tokenizer():
    data = dataset.load_data()
    hindi_sentences   = [x['translation'][config.DST_LANG] for x in data['train']]
    trg_token= train_tokenizer(hindi_sentences, lang=config.DST_LANG)
    return trg_token
    

