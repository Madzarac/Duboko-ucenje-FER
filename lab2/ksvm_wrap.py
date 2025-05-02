import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from data import graph_surface, graph_data, sample_gmm_2d
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score


class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.model = SVC(kernel='rbf', C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.model.fit(X, Y_)
        self.support = self.model.support_
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_scores(self, X):
        return self.model.decision_function(X)

if __name__ == "__main__":
    np.random.seed(100)
    X, Y_ = sample_gmm_2d(6, 2, 10)
    
    svm = KSVMWrap(X, Y_)
    Y_pred = svm.predict(X)
    
    accuracy = accuracy_score(Y_, Y_pred)
    precision = precision_score(Y_, Y_pred, average='macro')
    recall = recall_score(Y_, Y_pred, average='macro')
    avg_precision = average_precision_score(Y_, svm.get_scores(X))
    
    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"average precision: {avg_precision:.4f}")
    
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(lambda x: svm.predict(x), bbox, offset=0.5)
    
    graph_data(X, Y_, Y_pred, special=svm.support)
    plt.show()