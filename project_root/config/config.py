from transformers import AutoTokenizer
SRC_LANG_TOKENIZER_MODEL = AutoTokenizer.from_pretrained("google-T5/T5-base")

DATASET_PATH = "/home/thasin/class-projects/LLM-project/project_root/dataset/Sentence pairs in English-Hindi - 2025-02-13.tsv"

DATASET_SAVE = "dataset"

# Src_sentece = 'SrcSentence'

COLUMN_NAMES = ["SrcSentenceID","SrcSentence","DstSentenceID","DstSentence"]

SRC_LANG = "english"

DST_LANG = "hi"

SRC_LANG_VOCAB_FILENAME = "src_vocabulary.pkl"

DST_LANG_VOCAB_FILENAME = "dst_vocabulary.pkl"
 
SAVED_VOCAB_DIR = "encoding"

EXTRA_TOKEN_DICT = {"<PAD>":0,"<SOS": 1, "<EOS>": 2}

LONGEST_SRC_LANG_LENGTH = 68
    
LONGEST_DST_LANG_LENGTH = 68

EMBEDING_DIMS = 32

OPTIMIZER = "adam"

LOSS = "categorical_crossentorpy"

HIDDEN_LAYERS = 1

TRAINING_DATA_FRAC = 0.98

EPOCHS = 5

TESTING_DATA_FRAC = (1-TRAINING_DATA_FRAC)

MB_BATCH = 26

LEARNING_RATE = 0.001

