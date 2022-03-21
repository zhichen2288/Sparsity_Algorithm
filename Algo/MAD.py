import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import statistics

import torch
import torch.nn as nn


def find_mad_feature(features):
    def MAD(seq):
        deviation_list = []
        med_seq = statistics.median(seq)

        for i in seq:
            each_deviation = abs(i - med_seq)
            deviation_list.append(each_deviation)

        med_diviation = statistics.median(deviation_list)

        return med_diviation

    mad_feature = []

    for i in range(features.shape[1]):
        feature_seq = features.iloc[:, i]
        feature_seq = feature_seq.values

        each_mad_feature = MAD(feature_seq)
        mad_feature.append(each_mad_feature)

    mad_feature = torch.tensor(mad_feature)

    mad_feature = mad_feature.view(1, mad_feature.shape[0])

    return mad_feature




def find_CF_MAD(model, data, mask, mad_feature, is_df):
    if is_df:
        data = data.values
        data = data.astype(np.float32)
        data = torch.from_numpy(data)

        x0 = data.view(1, data.shape[1])
    else:
        feature, label = data
        x0 = feature.view(1, feature.shape[0])


    # Define Loss
    criterion = nn.CrossEntropyLoss()


    def proximity(original_input, counterfactual, mad_feature):
        distance = (abs(counterfactua l -original_input ) *( 1 /(mad_feature + 0.0000000001))).sum()
        return distance


    # Set Mask
    mask = torch.from_numpy(mask)

    epoch_number = 100
    lambda_list = [0.1, 1, 5, 10, 50, 100, 200, 500, 1000, 10000, 100000]
    learning_rate = 0.03


    # Predict x0
    y_pred_original = model(x0)
    _, prediction_original = torch.max(y_pred_original, 1)


    # Set Desired Label
    if prediction_original != 0:
        desired_label = prediction_original - 1

    elif prediction_original == 0:
        ret_0 = torch.tensor([0])
        return ret_0, ret_0, ret_0, 0


    # Loop
    for lambda_i in lambda_list:
        # dx_gd = torch.rand(1, x0.shape[1])
        ########
        dx_gd = torch.zeros(1, x0.shape[1])
        # ^^^^^^^

        dx_gd.requires_grad = True

        for epoch in range(epoch_number):
            dx_gd.data *= mask
            x1 = x0 + dx_gd
            y_pred_desired = model(x1)
            loss = lambda_ i *criterion(y_pred_desired, desired_label) + proximity(x0, x1, mad_feature)
            loss.backward()

            with torch.no_grad():
                dx_gd -= learning_rat e *dx_gd.grad


        # Test If We Get The Desired Label
        dx_gd.data *= mask
        x1 = x0 + dx_gd
        y_pred_desired = model(x1)
        _, prediction_desired = torch.max(y_pred_desired, 1)

        if prediction_desired == desired_label:
            break


    if prediction_desired != desired_label:
        ret_0 = torch.tensor([0])
        return ret_0, ret_0, ret_0, 0


    return x0, x1, dx_gd, lambda_i