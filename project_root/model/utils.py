from model import layers, transformer
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from Preprocessing import tokenizer

src_token = tokenizer.src_tokenizer()
trg_token = tokenizer.trg_tokenizer()


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
