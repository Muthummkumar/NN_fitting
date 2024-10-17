import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Define models
class SimpleNet(nn.Module):  # Underfitting
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc1(x)
        return x

class MediumNet(nn.Module):  # Optimal fit
    def __init__(self):
        super(MediumNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ComplexNet(nn.Module):  # Overfitting
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Train function
def train_model(model, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss = []
    test_loss = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(trainloader))
        
        # Evaluate on test data
        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        test_loss.append(running_loss / len(testloader))

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss[-1]}, Test Loss: {test_loss[-1]}')

    return train_loss, test_loss

# Plot function
def plot_losses(train_loss, test_loss, title):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Train models
simple_net = SimpleNet()
medium_net = MediumNet()
complex_net = ComplexNet()

print("Training SimpleNet (Underfitting)...")
train_loss_simple, test_loss_simple = train_model(simple_net)

print("\nTraining MediumNet (Optimal Fit)...")
train_loss_medium, test_loss_medium = train_model(medium_net)

print("\nTraining ComplexNet (Overfitting)...")
train_loss_complex, test_loss_complex = train_model(complex_net)

# Plot losses for each model
plot_losses(train_loss_simple, test_loss_simple, 'Underfitting: SimpleNet')
plot_losses(train_loss_medium, test_loss_medium, 'Optimal Fit: MediumNet')
plot_losses(train_loss_complex, test_loss_complex, 'Overfitting: ComplexNet')

