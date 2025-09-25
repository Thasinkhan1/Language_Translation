from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from project_root.config import config
from project_root.data import dataset,utils



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
    

#for testing pipeline
# from tokenizers import Tokenizer

# def test_tokenizer():
#     en_tok = Tokenizer.from_file(config.src_tokenizer)
#     hi_tok = Tokenizer.from_file(config.trg_tokenizer)

#     sample_text = "Hello world"
#     print(sample_text)
#     encoded = en_tok.encode(sample_text)
#     print("Encoded:", encoded.ids)
#     print("Decoded:", en_tok.decode(encoded.ids))
# # Load train, val, test batches
#     train_loader = utils.train_loader()
#     val_loader = utils.val_loader()
#     test_loader = utils.test_loader()
    
#     # Take one batch from train
#     src_batch, trg_batch = next(iter(train_loader))
    
#     print("Source batch shape:", src_batch.shape)
#     print("Target batch shape:", trg_batch.shape)
#     print("Example source IDs:", src_batch[0][:20])
#     print("Example target IDs:", trg_batch[0][:20])
    

# if __name__ == "__main__":
#     test_tokenizer()
