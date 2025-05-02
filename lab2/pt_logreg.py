import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data import graph_surface, graph_data
import data


def to_one_hot(Y, C):
    return torch.eye(C)[Y]

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
         - D: dimensions of each datapoint 
         - C: number of classes
        """
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        super().__init__()
        self.W = nn.Parameter(torch.randn(D, C) * 0.01)
        self.b = nn.Parameter(torch.zeros(1, C))
    
    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        return torch.softmax(torch.mm(X, self.W) + self.b, dim=1)
    
    def get_loss(self, X, Yoh_, param_lambda=0.01):
        # formulacija gubitka
        #   koristiti: torch.log, torch.exp, torch.sum
        #   pripaziti na numerički preljev i podljev
        P = self.forward(X)
        loss = -torch.mean(torch.sum(Yoh_ * torch.log(P + 1e-9), dim=1))
        loss += param_lambda * torch.norm(self.W) ** 2
        return loss

def train(model, X, Yoh_, param_niter=1000, param_delta=0.5, param_lambda=0.01):
    """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
    """
    # inicijalizacija optimizatora
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    # petlja učenja
    # ispisujte gubitak tijekom učenja
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print(f'Iteration {i}: Loss {loss.item():.6f}')

def eval(model, X):
    """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
    """
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    with torch.no_grad():
        return model(X).detach().numpy()

def compute_metrics(Y, Y_):
    C = np.max(Y_) + 1
    conf_matrix = np.zeros((C, C), dtype=int)
    for i in range(len(Y_)):
        conf_matrix[Y_[i], Y[i]] += 1
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    return accuracy, conf_matrix, precision, recall

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    X_np, Y_np = data.sample_gauss_2d(3, 100)
    
    # ulaz je potrebno pretvoriti u torch.Tensor
    # instanciraj podatke X i labele Yoh_
    X = torch.tensor(X_np, dtype=torch.float32)
    Yoh_ = to_one_hot(torch.tensor(Y_np, dtype=torch.long), 3)
    
    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])
    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X, Yoh_, 1000, 0.5)
    
    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    Y_pred = np.argmax(probs, axis=1)
    
    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, conf_matrix, precision, recall = compute_metrics(Y_pred, Y_np)
    print(f'Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:\n', conf_matrix)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    
    # iscrtaj rezultate, decizijsku plohu
    bbox = (np.min(X_np, axis=0), np.max(X_np, axis=0))
    graph_surface(lambda x: np.argmax(eval(ptlr, torch.tensor(x, dtype=torch.float32)), axis=1), bbox, offset=0.5)
    graph_data(X_np, Y_np, Y_pred)
    plt.show()
