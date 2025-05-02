import torch
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out_3'
BATCH_SIZE = 50
EPOCH = 8

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[1.0])
])

def create_mnist_loaders(): 
  train_dataset = MNIST(DATA_DIR, transform=transform)
  train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [55000, 5000])
  test_dataset = MNIST(DATA_DIR, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
  return train_loader, valid_loader, test_loader

def draw_conv_filters(epoch, layer):
  weights = layer.weight.squeeze()
  weights = weights.detach().cpu().numpy()
  weights =  (weights - weights.min()) / weights.max()
  for i in range(16):
    plt.subplot(2, 8, i+1)
    image = weights[i, :, :]
    plt.imshow(image)
    plt.title(i)
    plt.xticks([])
    plt.yticks([])
  plt.savefig(SAVE_DIR/f"epoch{epoch+1}.png")
  plt.show()
  
 
class NewModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=0, bias=True)
    self.maxpool1= nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0, bias=True)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(in_features=32*4*4, out_features=512, bias=True)
    self.fc2 = nn.Linear(in_features=512, out_features=10, bias=True)
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc2:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc2.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.maxpool1(h)
    h = torch.relu(h)
    
    h = self.conv2(h)
    h = self.maxpool2(h)
    h = torch.relu(h)

    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc2(h)
    return logits
  
  
def train(model, train_loader, valid_loader, optimizer, scheduler, criterion):
  epochs = EPOCH
  for epoch in range(epochs):
    epoch_batch_steps = []
    epoch_batch_losses = []
    losses_train = 0
    corrects_train = 0
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs  = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs , labels)
        loss.backward()
        optimizer.step()
        losses_train += loss.item() * inputs.size(0)
        corrects_train += torch.sum(preds == labels.data)
        if i % 5 == 0:
          epoch_batch_steps.append(i*BATCH_SIZE)
          epoch_batch_losses.append(loss.item())
          print(f"epoch {epoch+1}, step {i * BATCH_SIZE}/{len(train_loader.dataset)}, batch loss = {loss.item():.2f}")
        
    model.eval()
    losses_valid = 0
    corrects_valid = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)
          losses_valid += loss.item() * inputs.size(0)
          corrects_valid += torch.sum(preds == labels.data)

    accuracy_train = corrects_train / len(train_loader.dataset)
    accuracy_valid = corrects_valid / len(valid_loader.dataset)
    loss_train = losses_train / len(train_loader.dataset)
    loss_valid = losses_valid / len(valid_loader.dataset)
    print(f"Train accuracy = {accuracy_train:.2f}")
    print(f"Train avg loss = {loss_train:.2f}")
    print(f"Validation accuracy = {accuracy_valid:.2f}")
    print(f"Validation avg loss = {loss_valid:.2f}")
        
    with (SAVE_DIR / "train_stats.txt").open(mode="a") as f:
        f.write(f"{accuracy_train},{loss_train}\n")
    with (SAVE_DIR / "valid_stats.txt").open(mode="a") as f:
        f.write(f"{accuracy_valid},{loss_valid}\n")
        
    plt.plot(epoch_batch_steps,epoch_batch_losses)
    plt.show()

    draw_conv_filters(epoch, model.conv1)
    scheduler.step()
        
        
def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss  = 0
    correct = 0
    print()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data)
    running_loss  /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"Test accuracy = {accuracy:.2f}")
    print(f"Test avg loss = {running_loss:.2f}") 
    
  
  
if __name__ == '__main__':
  train_loader, valid_loader, test_loader = create_mnist_loaders()

  model = NewModel()
  print(model.parameters)
  draw_conv_filters(0, model.conv1)

  #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-1)
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.1)
  def lr_lambda(epoch):
    if epoch < 2:
        return 1.0
    elif epoch < 4:
        return 0.1
    elif epoch < 6:
        return 0.01
    else:
        return 0.001

  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
  criterion = torch.nn.CrossEntropyLoss()

  train(model, train_loader, valid_loader, optimizer, scheduler, criterion)
  evaluate(model, test_loader, criterion)