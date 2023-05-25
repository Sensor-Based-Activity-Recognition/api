# 3rd Party Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DataModule(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.data = [torch.tensor(image[1]).float() for image in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        # global average pooling layer
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # Fully connected layer 1
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        # softmax layer
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=6), nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(-1, 128)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Runner:
    def __init__(self, model_path):
        # Load model
        self.model = CNN()

        # Load checkpoint from model.pkt
        checkpoint = torch.load(model_path)

        # Load model state dict
        self.model.load_state_dict(checkpoint["state_dict"])

        # Set model to eval mode
        self.model.eval()

    def run(self, data):
        activities = [
            "Sitzen",
            "Laufen",
            "Velofahren",
            "Rennen",
            "Stehen",
            "Treppenlaufen",
        ]
        data = DataModule(data)

        results = {}
        for i, spectrogram in enumerate(data):
            # predict
            result = self.model(spectrogram.unsqueeze(0))
            # get softmax probabilities
            result = torch.softmax(result, dim=1).detach().numpy().round(3)
            # add to results
            results[i] = {
                activities[j]: str(result[0][j]) for j in range(len(activities))
            }

        return results
