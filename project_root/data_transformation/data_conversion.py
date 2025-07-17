from config import config
from transformers import AutoTokenizer # type: ignore
from data_loading import data_loading
from indicnlp.tokenize import indic_tokenize # type: ignore

def tokenize_src_language(data):
    
    tokenizer = config.SRC_LANG_TOKENIZER_MODEL
    
    data[config.COLUMN_NAMES[1]] = data[config.COLUMN_NAMES[1]].apply(tokenizer.tokenize)
    
    return data


def tokenize_dst_language(data):
    
    data[config.COLUMN_NAMES[3]]=data[config.COLUMN_NAMES[3]].apply(lambda x: indic_tokenize.trivial_tokenize(x,lang=config.DST_LANG))    

    return data


# if __name__ == "__main__":
#     tokenize_src_language(data_loading.load_data())
#     tokenize_dst_language(data_loading.load_data())