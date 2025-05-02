import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import math
import skimage.io as ski_io


# priprema i ucitavanje podataka
def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

DATA_DIR = 'datasets/CIFAR/batches/'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, f'data_batch_{i}'))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

# normalizacija
train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

train_x = train_x.transpose(0, 3, 1, 2)
valid_x = valid_x.transpose(0, 3, 1, 2)
test_x = test_x.transpose(0, 3, 1, 2)

train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
valid_dataset = TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y))
test_dataset  = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=50)
test_loader  = DataLoader(test_dataset, batch_size=50)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        dummy_input = torch.zeros(1, 3, 32, 32)
        out = self.pool(F.relu(self.conv1(dummy_input)))
        out = self.pool(F.relu(self.conv2(out)))
        self.flat_dim = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# evaluacija
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, average=None, zero_division=0)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Conf matrix:\n{cm}")
    print(f"Precision: {np.round(precision, 2)}")
    print(f"Recall: {np.round(recall, 2)}\n")

    return accuracy


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = w[:,:,:,i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    img_uint8 = (img * 255).astype(np.uint8)
    ski_io.imsave(os.path.join(save_dir, filename), img_uint8)

# crta netocno klasificirane slike   
def draw_image(img, mean, std):
    img = img.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    ski_io.imshow(img)
    ski_io.show()

# Treniranje
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

SAVE_DIR = 'cifar_filtri'
draw_conv_filters(0, 0, model.conv1.weight.detach().cpu().numpy(), SAVE_DIR)

num_epochs = 10
train_losses = []
valid_accuracies = []
lrs = []
train_accuracies = []
valid_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    lrs.append(optimizer.param_groups[0]['lr'])
    # evaluacija
    print(f"\nEpoch {epoch+1}    Loss: {train_loss:.4f}")
    print("Train:")
    train_acc = evaluate(model, train_loader, device)
    print("Val:")
    val_acc = evaluate(model, valid_loader, device)
    # pohrana rezultata
    train_accuracies.append(train_acc)
    valid_accuracies.append(val_acc)
    valid_losses.append(criterion(model(torch.tensor(valid_x, device=device).float()).detach(), torch.tensor(valid_y, device=device).long()).item())

    scheduler.step()
    draw_conv_filters(epoch+1, 0, model.conv1.weight.detach().cpu().numpy(), SAVE_DIR)

# vizualizacija
plt.figure(figsize=(12, 8))
# Cross-entropy loss
plt.subplot(2, 2, 1)
plt.plot(train_losses, 'm-', label='train')
plt.plot(valid_losses, 'c-', label='validation')
plt.title("Cross-entropy loss", fontsize=12)
plt.legend()
plt.grid()
# Average class accuracy
plt.subplot(2, 2, 2)
plt.plot(train_accuracies, 'm-', label='train')
plt.plot(valid_accuracies, 'c-', label='validation')
plt.title("Average class accuracy", fontsize=12)
plt.legend()
plt.grid()
# Learning rate
plt.subplot(2, 2, 3)
plt.plot(lrs, 'm-', label='learning_rate')
plt.title("Learning rate", fontsize=12)
plt.legend()
plt.grid()
# Donji desni je prazan kao na slici
plt.subplot(2, 2, 4)
plt.axis('off')
plt.tight_layout()
plt.show()


# 20 netočno klasificiranih slika s najvećim gubitkom
model.eval()
incorrect = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        losses = F.cross_entropy(outputs, labels.long(), reduction='none')

        for i in range(inputs.size(0)):
            if preds[i] != labels[i]:
                incorrect.append({
                    'image': inputs[i].cpu().numpy(),
                    'true_label': labels[i].item(),
                    'probs': probs[i].cpu().numpy(),
                    'loss': losses[i].item()
                })



# sortiranje po gubitku
incorrect.sort(key=lambda x: x['loss'], reverse=True)
top20 = incorrect[:20]

fig, axes = plt.subplots(4, 5, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    entry = top20[i]
    img = entry['image']
    true_label = entry['true_label']
    probs = entry['probs']
    top3 = np.argsort(probs)[::-1][:3]

    # uklanja normalizaciju
    img = img.transpose(1, 2, 0)  # CxHxW u HxWxC
    img = img * data_std + data_mean
    img = np.clip(img, 0, 255).astype(np.uint8)

    ax.imshow(img)
    ax.set_title(f"T:{true_label} | P:{top3[0]},{top3[1]},{top3[2]}", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()