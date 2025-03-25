from config import config
from data_loading import data_loading
import torch
from data_transformation import data_conversion
import pandas as pd


def English_to_numeric(data):
    
    tokenizer = config.SRC_LANG_TOKENIZER_MODEL
    data[config.COLUMN_NAMES[1]] = data[config.COLUMN_NAMES[1]].apply(tokenizer.convert_tokens_to_ids)
    
    print(data.head())


def convert_hindi_tokens_to_ids(data):
    Vd = set()
    
    for tokenized_hindi_sentence in data[config.COLUMN_NAMES[3]]:
        Vd.update(tokenized_hindi_sentence)

    hindi_vocab = {token: idx + 3 for idx, token in enumerate(Vd)}
    hindi_vocab[list(config.EXTRA_TOKEN_DICT.keys())[0]] = 0
    hindi_vocab[list(config.EXTRA_TOKEN_DICT.keys())[1]] = 1
    hindi_vocab[list(config.EXTRA_TOKEN_DICT.keys())[2]] = 2
    
    def tokenize_sentence(sentence):
        return [hindi_vocab[token] for token in sentence if token in hindi_vocab]

    data[config.COLUMN_NAMES[3]] = data[config.COLUMN_NAMES[3]].apply(tokenize_sentence)
    
    return data



# if __name__ == "__main__":
#     data1 = data_loading.load_data()
#     after_tok1 = data_conversion.tokenize_src_language(data1)
#     after_tok2 = data_conversion.tokenize_dst_language(data1)
#     English_to_numeric(after_tok1)
#     convert_hindi_tokens_to_ids(after_tok2)

#working good