from project_root.model import layers, transformer
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from project_root.config import config
from project_root.data import utils

src_token = Tokenizer.from_file(config.src_tokenizer)
trg_token = Tokenizer.from_file(config.trg_tokenizer)

# src_token = Tokenizer.from_file("en_tokenizer.json")
# trg_token = Tokenizer.from_file("hi_tokenizer.json")

INPUT_DIM = src_token.get_vocab_size()    # English vocab size
OUTPUT_DIM = trg_token.get_vocab_size()    # Hindi vocab size
EMBED_DIM = 256
HIDDEN_DIM = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attn = layers.Attention(HIDDEN_DIM)
enc = layers.Encoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM)
dec = layers.Decoder(OUTPUT_DIM, EMBED_DIM, HIDDEN_DIM, attn)

def load_model():
    model = transformer.Seq2Seq(enc, dec, device).to(device)
    return model

if __name__ == "__main__":
    train_loader = utils.train_loader()
    src_batch, trg_batch = next(iter(train_loader))
    src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
    model = load_model()
    
    with torch.no_grad():
         output = model(src_batch, trg_batch[:,:-1])
    print("Output shape:", output.shape)  