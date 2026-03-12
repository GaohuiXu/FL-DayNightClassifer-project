import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ==========================================
# Define a simple CNN model (using as daytime / nighttime classifer)
# ==========================================

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        super(SimpleCNN, self).__init__()
        # input: 3 channels(RGB), output: 16 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # input: 16 channels, output: 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Using MaxPool for 2 times，128x128 image is converted to 32x32 
        # Fully connected layer: 32 feature maps * 32 width * 32 height = 32768
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2) # output 2 classes: 0(Day) or 1(night)

        self.activate = nn.ReLU()

    def forward(self, x):
        # TODO
        x = self.pool(self.activate(self.conv1(x)))
        x = self.pool(self.activate(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = self.activate(self.fc1(x))
        x = self.fc2(x)

        return x

# ==========================================
# Federated Learning Parameter Conversion Tool
# ==========================================

# Sets the parameters of the model
def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)

# Retrieves the parameters from the model
def get_parameters(net):
    ndarrays = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]
    return ndarrays

# ==========================================
# Standard Pytorch Training and Testing Loop
# ==========================================

def train(net, train_loader, epochs, device, lr):
    """Standard Pytorch Training"""
    # TODO
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    total_loss = 0.0
    total_samples = 0

    net.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss

def eval(net, testloader, device):
    """Standard Pytorch Evaluation"""
    # TODO
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    net.eval()
    with torch.no_grad(): 
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    return loss, accuracy