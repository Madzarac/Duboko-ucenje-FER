import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers,
                 embeddings=None, freeze=True, bidirectional=False):
        super(GRUModel, self).__init__()
        self.bidirectional = bidirectional
        self.embeddings = embeddings
        
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
        n_features = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embeddings(x)
        embedded = embedded.permute(1, 0, 2)

        output, hidden = self.rnn(embedded)

        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        x = self.fc1(last_hidden)
        x = F.relu(x)
        x = self.fc2(x)

        return x.squeeze(1)
