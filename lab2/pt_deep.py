import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data import graph_surface, graph_data, sample_gmm_2d, sample_gauss_2d
from sklearn.metrics import accuracy_score, precision_score, recall_score

class PTDeep(nn.Module):
    def __init__(self, layer_dims, activation_fn=torch.sigmoid):
        super().__init__()
        self.activation_fn = activation_fn
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
                                         for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:])])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1, out_dim)) for out_dim in layer_dims[1:]])
    
    def forward(self, X):
        out = X
        for i in range(len(self.weights) - 1):
            out = self.activation_fn(torch.mm(out, self.weights[i]) + self.biases[i])
        return torch.softmax(torch.mm(out, self.weights[-1]) + self.biases[-1], dim=1)
    
    def get_loss(self, X, Yoh_, param_lambda=1e-4):
        P = self.forward(X)
        loss = -torch.mean(torch.sum(Yoh_ * torch.log(P + 1e-9), dim=1))
        loss += param_lambda * sum(torch.norm(W) ** 2 for W in self.weights)
        return loss
    
    def count_params(self):
        total_params = 0
        for name, param in self.named_parameters():
            print(f'{name}: {param.shape}, num params: {param.numel()}')
            total_params += param.numel()
        print(f'Total number of parameters: {total_params}')

def train(model, X, Yoh_, param_niter=10000, param_delta=0.1, param_lambda=1e-4):
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 1000 == 0:
            print(f'Iteration {i}: Loss {loss.item():.6f}')

def eval(model, X):
    with torch.no_grad():
        return model(X).detach().numpy()
    
def compute_metrics(Y_true, Y_pred):
    accuracy = accuracy_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred, average='macro', zero_division=0)
    recall = recall_score(Y_true, Y_pred, average='macro', zero_division=0)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

if __name__ == "__main__":
    np.random.seed(100)
    #torch.manual_seed(45)
    C = 2
    
    #X_np, Y_np = sample_gauss_2d(3, 100)
    X_np, Y_np = sample_gmm_2d(6, C, 10)
    X = torch.tensor(X_np, dtype=torch.float32)
    Yoh_ = torch.eye(C)[torch.tensor(Y_np, dtype=torch.long)]
    
    model = PTDeep([2, 10, 10, C], activation_fn=torch.relu)
    model.count_params()
    train(model, X, Yoh_, param_niter=10001, param_delta=0.1, param_lambda=1e-4)
    
    probs = eval(model, X)
    Y_pred = np.argmax(probs, axis=1)
    
    compute_metrics(Y_np, Y_pred)
    
    bbox = (np.min(X_np, axis=0), np.max(X_np, axis=0))
    graph_surface(lambda x: np.argmax(eval(model, torch.tensor(x, dtype=torch.float32)), axis=1), bbox, offset=0.5)
    graph_data(X_np, Y_np, Y_pred)
    plt.show()