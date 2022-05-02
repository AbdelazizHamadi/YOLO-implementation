import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as f
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create simple network

class NN(nn.Module):

    # define architecture
    def __init__(self, input_size, num_classes):  # 28 * 28 = (784)
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    # define how the data is processed (activation functions)
    def forward(self, x):
        x = nn.LeakyReLU(self.fc1(x))
        x = self.fc2(x)

        return x


class CNNBlock(nn.Module):

    def __init__(self, in_channels=3, num_classes=2, **kwargs):
        super(CNNBlock, self).__init__()
        self.in_channels = in_channels
        self.Conv1 = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(32, num_classes)

    def forward(self, data):
        #data = data.flatten(start_dim=1)
        data = nn.functional.softmax(self.Conv1(data))
        print(data.shape)
        data = self.maxpool(data)
        print(data.shape)
        data = nn.LeakyReLU(self.Conv1(data))
        #print(data.shape)

        data = nn.LeakyReLU(self.fc1(data))
        print(data.shape)

        return data


data = torch.randn(64, 3, 28, 28)
data = data.reshape(data.shape[0], -1)

model_cnn = CNNBlock(28 * 28, 1)

model = NN(784, 10)
print(model)


class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super(Net, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # pooling
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # activation
        self.relu = nn.ReLU()

    def forward(self, data):

            data = self.relu(self.conv1(data))
            data = self.relu(self.maxpool(data))
            data = self.relu(self.conv2(data))
            data = self.relu(self.maxpool(data))
            data = data.flatten(start_dim=1)
            data = self.relu(self.fc1(data))
            data = self.relu(self.fc2(data))
            data = self.relu(self.fc3(data))

            return data