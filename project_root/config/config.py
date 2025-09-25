
DATASET_PATH = "cfilt/iitb-english-hindi"

DATASET_SAVE = "dataset"

# Src_sentece = 'SrcSentence'

SRC_LANG = "en"

DST_LANG = "hi"

SRC_LANG_VOCAB_FILENAME = "src_vocabulary.pkl"

DST_LANG_VOCAB_FILENAME = "dst_vocabulary.pkl"
 
SAVED_VOCAB_DIR = "encoding"

EXTRA_TOKEN_LIST = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

VOCAB_SIZE = 30000

LONGEST_SRC_LANG_LENGTH = 68
    
LONGEST_DST_LANG_LENGTH = 68

EMBEDING_DIMS = 32

OPTIMIZER = "adam"

LOSS = "categorical_crossentorpy"

HIDDEN_LAYERS = 4

TRAINING_DATA_FRAC = 0.98

EPOCHS = 1

TESTING_DATA_FRAC = (1-TRAINING_DATA_FRAC)

MB_BATCH = 64

LEARNING_RATE = 1e-3



