import numpy as np
import matplotlib.pyplot as plt
from data import graph_surface, graph_data

def softmax(scores):
    expscores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return expscores / np.sum(expscores, axis=1, keepdims=True)

def logreg_train(X, Y_, param_niter=1000, param_delta=0.1):
    N, D = X.shape
    C = np.max(Y_) + 1
    
    W = np.random.randn(D, C)
    b = np.zeros((1, C))
    
    Y_onehot = np.eye(C)[Y_.flatten()]
    
    for i in range(param_niter):
        scores = np.dot(X, W) + b  # N x C
        probs = softmax(scores)  # N x C
        
        loss = -np.mean(np.sum(Y_onehot * np.log(probs), axis=1))
        
        if i % 10 == 0:
            print(f"iteration {i}: loss {loss}")
        
        dL_ds = probs - Y_onehot  # N x C
        grad_W = np.dot(X.T, dL_ds) / N  # D x C
        grad_b = np.mean(dL_ds, axis=0, keepdims=True)  # 1 x C
        
        W -= param_delta * grad_W
        b -= param_delta * grad_b
    
    return W, b

def logreg_classify(X, W, b):
    scores = np.dot(X, W) + b
    return softmax(scores)

def eval_perf_multi(Y, Y_):
    C = np.max(Y_) + 1
    conf_matrix = np.zeros((C, C), dtype=int)
    for i in range(len(Y_)):
        conf_matrix[Y_[i], Y[i]] += 1
    
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    
    return accuracy, conf_matrix, precision, recall

def logreg_decfun(W, b):
    def classify(X):
        return np.argmax(logreg_classify(X, W, b), axis=1)
    return classify

if __name__ == "__main__":
    import data
    np.random.seed(100)
    
    X, Y_ = data.sample_gauss_2d(3, 100)
    W, b = logreg_train(X, Y_)
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)
    
    accuracy, conf_matrix, precision, recall = eval_perf_multi(Y, Y_)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Precision:", precision)
    print("Recall:", recall)
    
    decfun = logreg_decfun(W, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0.5)
    
    graph_data(X, Y_, Y)
    
    plt.show()