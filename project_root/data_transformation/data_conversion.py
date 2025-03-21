from config import config
from transformers import AutoTokenizer
from data_loading import data_loading
from indicnlp import indic_tokenize

def tokenize_src_language(data):
    
    tokenizer = AutoTokenizer.from_pretrained(config.SRC_LANG_TOKENIZER_MODEL)
    
    data[config.COLUMN_NAMES[1]] = [config.COLUMN_NAMES[1]].apply(tokenizer.tokenize)
    
    return data


def tokenize_dst_language(data):
    
    data[config.COLUMN_NAMES[3]]=data[config.COLUMN_NAMES[3]].apply(lambda x: indic_tokenize.trivial_tokenize(x,lang=config.DST_LANG))    

    return data
