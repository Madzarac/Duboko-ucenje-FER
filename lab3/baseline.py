import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, embeddings):
        super(BaselineModel, self).__init__()
        self.embeddings = embeddings
        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embeddings(x) 
        x = torch.mean(x, dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x).squeeze(1)
        return x
