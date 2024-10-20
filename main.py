import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train() 
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # reseting gradient
            outputs = model(images)  # forward pass
            loss = criterion(outputs, labels)  # loss calculation
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # disable gradient 
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    

# visualization
def visualize_predictions(model, test_loader):
    model.eval()
    images, labels = next(iter(test_loader))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).numpy()  # convert tensor to image
        img = img * 0.5 + 0.5  # denormalize
        ax.imshow(img)
        ax.set_title(f'Pred: {classes[predicted[i]]}, True: {classes[labels[i]]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# each class accuracy
def evaluate_model_with_class_percentages(model, test_loader):
    model.eval()
    correct = [0] * 10
    total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for label in labels:
                total[label] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    correct[labels[i]] += 1


    for i in range(10):
        accuracy = 100 * correct[i] / total[i] if total[i] > 0 else 0
        print(f'Accuracy for class {classes[i]}: {accuracy:.2f}%')


train_model(model, train_loader, criterion, optimizer, num_epochs=10)

evaluate_model(model, test_loader)

visualize_predictions(model, test_loader)

evaluate_model_with_class_percentages(model, test_loader)


