from config import config
from data_transformation import data_conversion
from data_transformation import Conv_to_numeric
from data_loading import data_loading

import pandas as pd # type: ignore

def clean(data):
    
    sos_token_id = list(config.EXTRA_TOKEN_DICT.keys())[1]

    eos_token_id = list(config.EXTRA_TOKEN_DICT.keys())[2]

    def insert_sos_token_id(hindi_sentence_token_ids_list):
      return [sos_token_id] + hindi_sentence_token_ids_list

    def insert_eos_token_id(hindi_sentence_token_ids_list):
       return hindi_sentence_token_ids_list + [eos_token_id]


    data["DstSentenceInput"] = data[config.COLUMN_NAMES[3]].apply(insert_sos_token_id)
    data["DstSentenceLabel"] = data[config.COLUMN_NAMES[3]].apply(insert_eos_token_id)
    
    data.drop(labels=[config.COLUMN_NAMES[0],config.COLUMN_NAMES[2],config.COLUMN_NAMES[3]],axis=1,inplace=True)
    
    return data


# if __name__ == "__main__":
#     data = data_loading.load_data()
#     after_tok = data_conversion.tokenize_dst_language(data)
#     encoded = Conv_to_numeric.convert_hindi_tokens_to_ids(after_tok)
#     clean(encoded)
