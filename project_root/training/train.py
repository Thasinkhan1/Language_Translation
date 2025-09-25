import torch
import os
from project_root.config import config
from project_root.data import data_loader
from project_root.model import utils
from project_root.training.evaluate import evaluate
from tokenizers import Tokenizer

src_token = Tokenizer.from_file(config.src_tokenizer)
pad_idx = src_token.token_to_id(config.EXTRA_TOKEN_LIST[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = utils.load_model()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx,label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

scaler = torch.cuda.amp.GradScaler()
accum_steps = 4  # simulate batch size 128
save_path = "checkpoints"
def train_model(num_epochs):
    best_val_loss = float("inf")
    train_data = utils.train_loader()
    val_data = utils.val_loader()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for i, (src, trg) in enumerate(train_data):
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(src, trg[:, :-1])  # shift target
                loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))

            loss = loss / accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

            # Print progress every 500 steps
            if (i + 1) % 500 == 0:
                print(f"Epoch {epoch+1}, Step {i+1}/{len(train_data)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_data)
        val_loss = evaluate(model, val_data, criterion, device)
        print(f"Epoch {epoch+1} Complete | Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, f"best_model_epoch{epoch+1}.pt"))
            print("Saved new best model!")


if __name__ == "__main__":
    train_model(num_epochs=1)