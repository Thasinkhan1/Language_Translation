from nltk.translate.bleu_score import sentence_bleu
from project_root.model import utils
from tokenizers import Tokenizer
from project_root.config import config
import torch

model = utils.load_model()

src_token = Tokenizer.from_file(config.src_tokenizer)
trg_token = Tokenizer.from_file(config.trg_tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load(config.SAVED_MODEL_PATH))
model.to(device)
def translate_sentence(sentence):
    model.eval()
    src_ids = [src_token.token_to_id("<SOS>")] + src_token.encode(sentence).ids + [src_token.token_to_id("<EOS>")]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    encoder_outputs, hidden = model.encoder(src_tensor)
    input = torch.tensor([trg_token.token_to_id("<SOS>")]).to(device)

    result = []
    for _ in range(50):
        output, hidden = model.decoder(input, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        if pred_token == trg_token.token_to_id("<EOS>"):
            break
        result.append(pred_token)
        input = torch.tensor([pred_token]).to(device)

    return trg_token.decode(result)


if __name__ == "__main__":
    print(translate_sentence(("how are you")))
