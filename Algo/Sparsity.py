import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()

        self.lin_start = nn.Linear(input_size, 400)
        # self.lin1 = nn.Linear(800, 400)
        self.lin_end = nn.Linear(400, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.lin_start(x)
        out = self.relu(out)

        # out = self.lin1(out)
        # out = self.relu(out)

        out = self.lin_end(out)

        return out


def sparsity(X0, sector, input_size, output_size, w, if_ignore=True):
    # deal with x0
    model = NeuralNetwork(input_size, output_size)
    model.load_state_dict(torch.load('./model/' + sector))

    X0 = X0.astype(np.float32)
    X0 = torch.from_numpy(X0)
    x0 = torch.clone(X0)
    x0 = x0.view(1, x0.shape[1])

    def proximity1(original_input):
        distance = abs(original_input).sum()
        return distance

    criterion = nn.CrossEntropyLoss()
    # predict xo
    y_pred_original = model(x0)

    # get information from y_pred
    _, predictions_original = torch.max(y_pred_original, 1)

    # define desired label
    if predictions_original != 0:
        desired_label = predictions_original - 1
    elif predictions_original == 0:
        r_0 = torch.tensor([[0]])
        return r_0, 0, r_0, r_0, r_0, r_0, r_0, r_0

    # set parameters
    lambda_list = [0.1, 1, 5, 10, 50, 100, 200, 500, 1000, 10000, 100000]
    epoch_number = 100
    learning_rate = 0.03
    W = torch.from_numpy(w)

    # gradient descent
    for lambda_i in lambda_list:
        x1 = torch.zeros(1, x0.shape[1])
        x1.requires_grad = True

        for epoch in range(epoch_number):
            x1_prime = x0 + x1
            y_pred_desired = model(x1_prime)
            loss = lambda_i * criterion(y_pred_desired, desired_label) + proximity1(x1)
            loss.backward()

            with torch.no_grad():
                x1 -= learning_rate * x1.grad

            x1.data *= W

        x1_prime = x0 + x1
        y_pred_desired = model(x1_prime)
        _, predictions_desired = torch.max(y_pred_desired, 1)

        if predictions_desired == desired_label:
            break

    if predictions_desired != desired_label:
        r_0 = torch.tensor([[0]])
        return r_0, 0, r_0, r_0, r_0, r_0, r_0, r_0

    # ratio part
    x1_middle = x1.clone()
    change_ratio = x1_middle / (x0 + torch.tensor(0.000000001))

    if if_ignore:
        change_ratio[abs(change_ratio) > 1000] = 0

    else:
        change_ratio[abs(change_ratio) > 1000] = 1

    # if ignore:
    #     change_ratio[change_ratio <  10000] = 0
    # else:
    #     change_ratio[change_ratio <  10000] = 1
    index_change = []
    for i in enumerate(change_ratio[0]):
        index_change.append(i)

    index_change_df = pd.DataFrame(index_change,
                                   columns=['index', 'change'])

    index_change_df.loc[:, 'change'] = index_change_df.loc[:, 'change'].abs()

    index_change_df = index_change_df.sort_values(['change'],
                                                  ascending=0)

    index_list = index_change_df.index

    index_change_dictionary = {}
    for i in index_list:
        index_change_dictionary[i] = x1_middle[0, i]

    keep_index = []
    for i in index_list:
        x1_Dan = x1_middle.clone()
        keep_index.append(i)

        for j in range(len(x1_middle[0])):
            if j not in keep_index:
                x1_Dan[0][j] = 0

        x1_prime_Dan = x0 + x1_Dan
        y_pred_Dan = model(x1_prime_Dan)

        _, prediction_Dan = torch.max(y_pred_Dan, 1)

        if prediction_Dan == desired_label:
            break

    return x0, lambda_i, y_pred_original, predictions_original, y_pred_Dan, prediction_Dan, x1_middle, x1_Dan