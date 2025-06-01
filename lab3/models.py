# rnn_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, num_layers, padding_idx,
                 embeddings=None, freeze=True, bidirectional=False, dropout=0.0):
        super(RNNModel, self).__init__()
        self.bidirectional = bidirectional
        self.embeddings = embeddings

        RNNClass = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }[rnn_type.lower()]

        self.rnn_type = rnn_type.lower()
        self.rnn = RNNClass(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=False,
            dropout=dropout
        )

        n_features = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embeddings(x)  # [B, T, D]
        embedded = embedded.permute(1, 0, 2)  # [T, B, D]

        output, hidden = self.rnn(embedded)

        if self.rnn_type == 'lstm':
            hidden = hidden[0]  # samo hidden state

        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        x = self.fc1(last_hidden)
        x = F.relu(x)
        x = self.fc2(x)

        return x.squeeze(1)
