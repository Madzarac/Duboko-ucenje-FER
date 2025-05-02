import torch
import torch.nn as nn
import torch.optim as optim

def analytical_gradients(X, Y, a, b):
    N = X.shape[0]
    dL_da = (-2 / N) * torch.sum(X * (Y - (a * X + b)))
    dL_db = (-2 / N) * torch.sum(Y - (a * X + b))
    return dL_da, dL_db

def train_linear_regression(X, Y, lr=0.01, n_iter=100):
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    optimizer = optim.SGD([a, b], lr=lr)
    
    for i in range(n_iter):
        Y_pred = a*X + b
        diff = Y - Y_pred
        loss = torch.mean(diff ** 2)
        loss.backward()
        
        dL_da_analytical, dL_db_analytical = analytical_gradients(X, Y, a, b)
        
        print(f'step: {i}, loss: {loss.item():.6f}')
        print(f'  PyTorch grad a: {a.grad.item():.6f}, Analytical grad a: {dL_da_analytical.item():.6f}')
        print(f'  PyTorch grad b: {b.grad.item():.6f}, Analytical grad b: {dL_db_analytical.item():.6f}')
        
        optimizer.step()
        optimizer.zero_grad()
    
    return a, b

if __name__ == "__main__":
    X = torch.tensor([1, 2, 3, 4, 5])
    Y = torch.tensor([3, 5, 6, 7, 8])
    train_linear_regression(X, Y)