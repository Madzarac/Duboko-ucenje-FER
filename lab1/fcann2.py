import numpy as np
import matplotlib.pyplot as plt
from data import graph_surface, graph_data, sample_gmm_2d

def reLu(X):
    """Aktivacijska funkcija, koristi se kod skrivenog sloja

    Arguments:
        X: podaci sa izlaza prvoj sloja

    Returns:
        X, ali pozitivni podaci ostaju isti, a negativni su zamijenjeni sa 0
    """
    return np.maximum(0, X)

def softmax(X):
    """Aktivacijska funkcija, koristi se na izlazu iz modela

    Arguments:
        X: podaci sa izlaza drugog sloja

    Returns:
        izlaz softmax funkcije
    """
    e_x = np.exp(X - np.max(X, axis=1, keepdims=True))  # numericka stabilnost, sprijecava velike eksopnentne vrijednosti
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def fcann2_train(X, Y_, param_niter=100000, param_delta=0.05, param_lambda=1e-3, hidden_dim=5):
    """Treniranje modela

    Arguments:
        X: ulazni podaci, dimenzija NxD
        Y_: stvarne oznake, vektor dimenzija Nx1
        param_niter: broj iteracija treniranja
        param_delta: stopa učenja (veličina koraka)
        param_lambda: regularizacijski parametar
        hidden_dim: broj neurona skrivenog sloja

    Returns:
        Slobodne parametre W1, b1, W2, b2
    """
  
    N, D = X.shape
    C = np.max(Y_) + 1
    
    # Inicijalizacija
    W1 = np.random.normal(loc=0, scale=1 / np.mean([D, hidden_dim]), size=(D, hidden_dim))
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.normal(loc=0, scale=1 / np.mean([hidden_dim, C]), size=(hidden_dim, C))
    b2 = np.zeros((1, C))
    
    # Treniranje    
    for i in range(param_niter):
        # Unaprijedna mreža
        s1 = np.dot(X, W1) + b1   # s1 = W1*x + b1
        h1 = reLu(s1)             # h1 = reLu(s1)
        s2 = np.dot(h1, W2) + b2  # s2 = W2*h1 + b2
        s2 -= s2.max()            # zastita od overflowa
        Y = softmax(s2)
        
        if i % 10000 == 0:
            loss = -np.mean(np.log(Y[range(N), Y_] + 1e-13)) + param_lambda * (np.linalg.norm(W1) + np.linalg.norm(W2))
            print(f"Iteration {i}: Loss {loss:.06f}")
        
        # Propagacija unatrag, izracun gradijenata
        Gs2 = Y
        Gs2[range(N), Y_] -= 1
        Gs2 /= N

        grad_W2 = np.dot(h1.T, Gs2) + param_lambda * W2   # (Pij - Yij) * h1.T + grad reg
        grad_b2 = np.sum(Gs2, axis=0, keepdims=True)      # Pij - Yij
        
        Gs1 = np.dot(Gs2, W2.T)   # prehodno * diagonalna[s1>0]
        Gs1[s1 < 0] = 0  
        
        grad_W1 = np.dot(X.T, Gs1) + param_lambda * W1
        grad_b1 = np.sum(Gs1, axis=0, keepdims=True)
        
        # update tezina
        W1 -= param_delta * grad_W1
        b1 -= param_delta * grad_b1
        W2 -= param_delta * grad_W2
        b2 -= param_delta * grad_b2
    
    return W1, b1, W2, b2

def fcann2_classify(X, W1, b1, W2, b2):
    """Provodi klasifikaciju primjera

    Arguments:
        X: podaci
        W1: težine prvog sloja
        b1: pomak prvog sloja
        W2: težine drugog sloja
        b2: pomak drugog sloja

    Returns:
        predviđene klase
    """
    S1 = np.dot(X, W1) + b1
    H1 = reLu(S1)
    S2 = np.dot(H1, W2) + b2
    return np.argmax(softmax(S2), axis=1)

def fcann2_decorator(W1, b1, W2, b2):
    """Omotač fcann2_classify

    Arguments:
        W1: težine prvog sloja
        b1: pomak prvog sloja
        W2: težine drugog sloja
        b2: pomak drugog sloja

    Returns:
        predviđene klase
    """
    def classify(X):
        return fcann2_classify(X, W1, b1, W2, b2)
    return classify

if __name__ == "__main__":
    np.random.seed(100)
    
    X, Y_ = sample_gmm_2d(6, 2, 10)
    W1, b1, W2, b2 = fcann2_train(X, Y_)
    Y = fcann2_classify(X, W1, b1, W2, b2)
    
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(fcann2_decorator(W1, b1, W2, b2), bbox, offset=0.5)
    
    graph_data(X, Y_, Y)
    plt.show()