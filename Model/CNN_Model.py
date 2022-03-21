import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def preprocess_data(features, rating, test_size=0.3, batch_size=100):
    # Combine Feature With Label
    df = pd.concat([rating, features], \
                   axis=1)

    # Split Training Set And Test Set
    x_training, x_test, y_training, y_test = train_test_split(df.iloc[:, :],
                                                              df.iloc[:, 0],
                                                              test_size=test_size,
                                                              random_state=42)

    # Create The Dataset For Training Set
    class TrainingSet(Dataset):

        def __init__(self):
            xy = x_training.values

            self.x = xy[:, 1:].astype(np.float32)
            self.y = xy[:, 0].astype(np.int64)

            self.x = torch.from_numpy(self.x)
            self.y = torch.from_numpy(self.y)

            self.n_samples = xy.shape[0]

        def __getitem__(self, index):
            return self.x[index], self.y[index]

        def __len__(self):
            return self.n_samples

    # Create An Object Of TrainingSet
    training_set = TrainingSet()

    # Create The Dataset For Test Set
    class TestSet(Dataset):

        def __init__(self):
            xy = x_test.values

            self.x = xy[:, 1:].astype(np.float32)
            self.y = xy[:, 0].astype(np.int64)

            self.x = torch.from_numpy(self.x)
            self.y = torch.from_numpy(self.y)

            self.n_samples = xy.shape[0]

        def __getitem__(self, index):
            return self.x[index], self.y[index]

        def __len__(self):
            return self.n_samples

    # Create An Object Of TestSet
    test_set = TestSet()

    # Create The Dataloader
    training_loader = DataLoader(dataset=training_set,
                                 batch_size=batch_size,
                                 shuffle=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=True)

    return training_set, test_set, training_loader, test_loader


class NeuralNetworkCNN(nn.Module):
    def __init__(self, input_dim, output_dim, sign_size, cha_input, cha_hidden, cha_increase_rate):
        super().__init__()
        hidden_size = sign_size * cha_input
        self.extend = nn.Linear(input_dim, hidden_size, bias=False)

        self.sign_size_1 = sign_size
        self.cha_input = cha_input

        self.conv1 = nn.Conv1d(cha_input,
                               cha_input * cha_increase_rate,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               groups=cha_input,
                               bias=False
                               )

        sign_size_2 = sign_size // 2
        self.ave_pool_1 = nn.AdaptiveAvgPool1d(output_size=sign_size_2)

        self.conv2 = nn.Conv1d(cha_input * cha_increase_rate,
                               cha_hidden,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.ave_pool_2 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        output_size = (sign_size // 4) * cha_hidden

        self.lin_final = nn.Linear(output_size,
                                   output_dim,
                                   bias=False)

    def forward(self, x):
        x = self.extend(x)
        x = x.reshape(x.shape[0], self.cha_input, self.sign_size_1)

        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.ave_pool_1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.ave_pool_2(x)

        x = self.flt(x)

        x = self.lin_final(x)

        return x


input_size = 294
output_size = 20

model = NeuralNetwork(input_size, output_size)


def train_model(model, training_loader, learning_rate, num_epoch):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = learning_rate)

    for epoch in range(num_epoch):

        for i, (feature, label) in enumerate(training_loader):
            y_pred = model(feature)
            loss = criterion(y_pred, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epoch+1)%100 == 0:
                print(f'''epoch: {epoch+1}/{num_epoch}, 
                          step: {i+1}/{len(training_loader)}, 
                          loss: {loss.item():.4f}
                       ''')


def test_model(model, test_loader):
    with torch.no_grad():
        n_correct = 0
        n_sample = 0

        for feature_t, label_t in test_loader:
            y_pred_t = model(feature_t)

            _, prediction = torch.max(y_pred_t, 1)

            n_correct += (prediction == label_t).sum().item()
            n_sample += label_t.shape[0]

        acc = n_correct/n_sample*100


    print(f"accuracy = {acc}")
