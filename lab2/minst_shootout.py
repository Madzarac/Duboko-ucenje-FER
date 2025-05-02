import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import random_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    dataset_root = '/tmp/mnist'
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)
    
    x_train, y_train = mnist_train.data.float().div(255.0), mnist_train.targets
    x_test, y_test = mnist_test.data.float().div(255.0), mnist_test.targets
    
    return x_train, y_train, x_test, y_test

def split_train_validation(x_train, y_train, validation_ratio=0.2):
    total_size = len(x_train)
    val_size = int(total_size * validation_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(list(zip(x_train, y_train)), [train_size, val_size])
    
    x_train, y_train = zip(*train_dataset)
    x_val, y_val = zip(*val_dataset)
    
    return torch.stack(x_train), torch.tensor(y_train), torch.stack(x_val), torch.tensor(y_val)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(784, 10)  # 784 ulaza (28x28 piksela) i 10 izlaza (klase 0-9)
    
    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc(x)
    
def train_simple_model(l2_reg):
    x_train, y_train, x_test, y_test = load_data()
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=l2_reg)
    
    epochs = 10
    for epoch in range(epochs):
        for i in range(0, x_train.shape[0], 64):
            images, labels = x_train[i:i+64], y_train[i:i+64]
            optimizer.zero_grad()
            outputs = model(images.view(-1, 784))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return model

# Funkcija za vizualizaciju težina
def visualize_weights(model, title):
    weights = model.fc.weight.detach().cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(weights[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Class {i}')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

class DeepNN(nn.Module):
    def __init__(self, layer_sizes):
        super(DeepNN, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x.view(x.size(0), -1))

# train za drugi i treci zadatak
def train_model(layer_sizes, x_train, y_train, x_val=None, y_val=None, epochs=20, lr=0.01, use_validation=False, patience=3, l2_reg=0.001):
    model = DeepNN(layer_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2_reg)
    train_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), batch_size=64, shuffle=True)
    
    if use_validation:
        val_loader = torch.utils.data.DataLoader(list(zip(x_val, y_val)), batch_size=64, shuffle=False)
    
    best_val_loss = float('inf')
    best_model = None
    losses = []
    val_losses = []
    patience_counter = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        
        if use_validation:
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Ako se gubitak nije poboljšao x puta, prekidamo trening
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    if use_validation and best_model:
        model.load_state_dict(best_model)
    
    return model, losses, val_losses if use_validation else None

# izracun metrika
def evaluate_model(model, x_test, y_test):
    test_loader = torch.utils.data.DataLoader(list(zip(x_test, y_test)), batch_size=64, shuffle=False)
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    
    return accuracy, precision, recall, all_preds, all_labels

def plot_errors(x_test, preds, labels):
    errors = np.where(np.array(preds) != np.array(labels))[0]
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    
    for i, ax in enumerate(axes.flat):
        idx = errors[i]
        ax.imshow(x_test[idx].numpy(), cmap='gray')
        ax.set_title(f'True: {labels[idx]}, Pred: {preds[idx]}')
        ax.axis('off')
    plt.show()

# za drugi i treci zadatak
def main(epochs, lr, use_validation, patience, l2_reg):
    x_train, y_train, x_test, y_test = load_data()
    
    if use_validation:
        x_train, y_train, x_val, y_val = split_train_validation(x_train, y_train, validation_ratio=0.2)
    
    results = {}
    
    for config in configs:
        print(f'Training model: {config}')
        if use_validation:
            model, losses, val_losses = train_model(config, x_train, y_train, x_val=x_val, y_val=y_val, epochs=epochs, lr=lr, use_validation=use_validation, patience=patience, l2_reg=l2_reg)
        else:
            model, losses, _ = train_model(config, x_train, y_train, x_val=None, y_val=None, epochs=epochs, lr=lr, use_validation=use_validation, patience=patience, l2_reg=l2_reg)
            val_losses = None
        
        accuracy, precision, recall, preds, labels = evaluate_model(model, x_test, y_test)
        results[str(config)] = (losses, val_losses, accuracy, precision, recall, preds, labels)
    
    for config, (losses, val_losses, _, _, _, _, _) in results.items():
        plt.plot(losses, label=f'Model {config} - Train Loss')
        if val_losses:
            plt.plot(val_losses, label=f'Model {config} - Validation Loss', linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    best_config = max(results, key=lambda x: results[x][2])
    _, _, _, _, _, preds, labels = results[best_config]
    plot_errors(x_test, preds, labels)
    
# mini-batch
def train_mb(layer_sizes, epochs=10, lr=0.01, batch_size=64, patience=3):
    x_train, y_train, x_test, y_test = load_data()
    
    indices = torch.randperm(len(x_train))
    val_size = len(x_train) // 5
    val_indices, train_indices = indices[:val_size], indices[val_size:]
    x_val, y_val = x_train[val_indices], y_train[val_indices]
    x_train, y_train = x_train[train_indices], y_train[train_indices]
    
    model = DeepNN(layer_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    losses = []
    val_losses = []
    
    for epoch in range(epochs):
        indices = torch.randperm(len(x_train))
        x_train_shuffled, y_train_shuffled = x_train[indices], y_train[indices]
        
        running_loss = 0.0
        for i in range(0, len(x_train), batch_size):
            images, labels = x_train_shuffled[i:i+batch_size], y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        avg_loss = running_loss / len(x_train)
        losses.append(avg_loss)
        
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break
    
    evaluate_model(model, x_test, y_test)
    return best_model, losses, val_losses

def train_adam(layer_sizes, epochs=10, lr=1e-4, batch_size=64, patience=3):
    x_train, y_train, x_test, y_test = load_data()
    
    indices = torch.randperm(len(x_train))
    val_size = len(x_train) // 5
    val_indices, train_indices = indices[:val_size], indices[val_size:]
    x_val, y_val = x_train[val_indices], y_train[val_indices]
    x_train, y_train = x_train[train_indices], y_train[train_indices]
    
    model = DeepNN(layer_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    losses = []
    val_losses = []
    
    for epoch in range(epochs):
        indices = torch.randperm(len(x_train))
        x_train_shuffled, y_train_shuffled = x_train[indices], y_train[indices]
        
        running_loss = 0.0
        num_batches = len(x_train) // batch_size
        
        for i in range(0, len(x_train), batch_size):
            images, labels = x_train_shuffled[i:i+batch_size], y_train_shuffled[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / num_batches
        avg_loss = running_loss
        losses.append(avg_loss)
        
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)
            
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
            
    evaluate_model(model, x_test, y_test)
    return best_model, losses, val_losses

def plot_loss(results):
    plt.figure(figsize=(8, 5))
    for config, (_, losses, val_losses) in results.items():
        plt.plot(losses, label=f'Train {config}')
        plt.plot(val_losses, label=f'Val {config}', linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def compute_initial_loss(layer_sizes):
    x_train, y_train, x_test, y_test = load_data()
    model = DeepNN(layer_sizes)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        outputs = model(x_test)
        loss = criterion(outputs, y_test).item()
    print(f'Initial loss (no training): {loss:.4f}')
    return loss

def SVM():
    dataset_root = '/tmp/mnist'
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data.numpy().reshape(-1, 28*28) / 255.0, mnist_train.targets.numpy()
    x_test, y_test = mnist_test.data.numpy().reshape(-1, 28*28) / 255.0, mnist_test.targets.numpy()

    print("Treniranje linearnog SVM-a...")
    svm_linear = SVC(kernel='linear', decision_function_shape='ovo')
    svm_linear.fit(x_train, y_train)
    y_pred_linear = svm_linear.predict(x_test)

    print("Treniranje jezgrenog SVM-a...")
    svm_rbf = SVC(kernel='rbf', decision_function_shape='ovo')
    svm_rbf.fit(x_train, y_train)
    y_pred_rbf = svm_rbf.predict(x_test)

    print("\nLINEARNI SVM")
    print(f"Točnost: {accuracy_score(y_test, y_pred_linear):.4f}")
    print(classification_report(y_test, y_pred_linear))

    print("\nJEZGREN SVM (RBF)")
    print(f"Točnost: {accuracy_score(y_test, y_pred_rbf):.4f}")
    print(classification_report(y_test, y_pred_rbf))
    
    
def task1(l2_reg=0.1):
    model = train_simple_model(l2_reg)
    visualize_weights(model, 'Težine')
    
def task2(epochs=10, lr=0.01, use_validation=False, patience=5, l2_reg=0.001):
    main(epochs, lr, use_validation, patience, l2_reg)
    
def task3(epochs=10, lr=0.01, use_validation=True, patience=5, l2_reg=0.001):
    main(epochs, lr, use_validation, patience, l2_reg)
    
def task4(config=[784, 100, 10], epochs=10, lr=0.01):
    results = {}
    print(f'Training model: {config}')
    model, losses, val_losses = train_mb(config, epochs, lr)
    results[str(config)] = (model, losses, val_losses)
    plot_loss(results)
    
    
def task5(config=[784, 100, 10], epochs=10, lr=0.01):
    results = {}
    print(f'Training model with config: {config}')
    model, losses, val_losses = train_adam(config, epochs=10, lr=1e-4)
    results[str(config)] = (model, losses, val_losses)
    plot_loss(results)

def task6(config=[784, 100, 10]):
    initial_loss = compute_initial_loss(config)
        
def task7():
    SVM()
    

if __name__ == "__main__":
    configs = [[784, 10], [784, 100, 10]]
    l2_reg = 0.00001
    
    #task1(l2_reg) # model konfiguracije [784, 10], matrice težina za svaku pojedinu znamenku, podesiv iznos regularizacije
    #task2() # pohrana i usporedba gubitka, ispis pokazatelja performansi
    #task3() # sve kao task2, ali postoji skup za validaciju
    #task4() # stohastički gradijentni spust
    #task5() # ADAM
    #task6() # gubitak slučajno incijaliziranog modela, bez učenja
    #task7() #SVM