import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))   # [batch, src_len, embed_dim]
        outputs, hidden = self.rnn(embedded)           # outputs: [batch, src_len, hidden_dim]
        return outputs, hidden
    
    
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hidden_dim]
        hidden = hidden[-1].unsqueeze(1)               # [batch, 1, hidden_dim]
        src_len = encoder_outputs.size(1)

        hidden = hidden.repeat(1, src_len, 1)          # [batch, src_len, hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)          # [batch, src_len]

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, attention, num_layers=1, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(hidden_dim + embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)                     # [batch, 1]
        embedded = self.dropout(self.embedding(input)) # [batch, 1, embed_dim]

        # Attention
        attn_weights = self.attention(hidden, encoder_outputs)   # [batch, src_len]
        attn_weights = attn_weights.unsqueeze(1)                 # [batch, 1, src_len]

        context = torch.bmm(attn_weights, encoder_outputs)       # [batch, 1, hidden_dim]

        rnn_input = torch.cat((embedded, context), dim=2)        # [batch, 1, embed+hidden]
        output, hidden = self.rnn(rnn_input, hidden)

        prediction = self.fc(torch.cat((output, context, embedded), dim=2).squeeze(1))
        return prediction, hidden
