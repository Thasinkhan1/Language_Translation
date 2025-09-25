
DATASET_PATH = "cfilt/iitb-english-hindi"

DATASET_SAVE = "dataset"

SRC_LANG = "en"

DST_LANG = "hi"
 
src_tokenizer = "project_root/tokenizers/en_tokenizer.json"
trg_tokenizer = "project_root/tokenizers/hi_tokenizer.json"

EXTRA_TOKEN_LIST = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

VOCAB_SIZE = 30000

SAVED_MODEL_PATH = "project_root/saved_model/model_weights.pth"

OPTIMIZER = "adam"

LOSS = "categorical_crossentorpy"

EPOCHS = 1

MB_BATCH = 64

LEARNING_RATE = 1e-3



