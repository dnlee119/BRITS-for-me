import os, sys, time, random
from models import brits
import util
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Function to Calculate Validation Loss
def validation_mae(val_df, imputations):
    n, sum_ = 0, 0
    validation_range = range(7799, 7967)
    num_list = list(validation_range)
    for i in num_list:
        k = abs(imputations[i][3] - val_df["value_1"].iloc[i - 7799])
        sum_ += k
        n += 1
    mae = sum_ / n
    return mae


# A function that returns normalized values to their original values.
def to_standard(df, weight_):
    length = len(df.columns) - 1
    scaler = StandardScaler()
    scaler = scaler.fit(df.drop("time", axis=1).to_numpy().reshape(-1, length))
    result = scaler.inverse_transform(weight_)
    result = result.reshape(-1, length)
    return result


# The actual Train function
def model_train(model, optim, data_iter, name):
    model.train()
    progress = tqdm(range(epoch))
    # Variables to store values in progress
    loss_graphic = []
    val_loss_graphic = []
    imputation = []
    n = 0
    #
    df = pd.read_csv("./data/" + name)
    val_df = pd.read_csv("./data/val_data.csv")
    att_col = df.columns
    for i in progress:
        n += 1
        total_loss = 0.0
        save_impute = []
        for idx, data in enumerate(data_iter):
            data = util.to_var(data)
            ret = model.run_on_batch(data, optim, i)
            total_loss += ret["loss"].item()
            save_impute.append(ret['imputations'].data.cpu().numpy())
        epoch_loss = (total_loss / len(data_iter))
        imputation = np.concatenate(save_impute, axis=0)
        result = to_standard(df, imputation)
        loss_graphic.append(epoch_loss)
        val_loss_graphic.append(validation_mae(val_df, result))
        if n >= 3:
            if val_loss_graphic[-1] < min(val_loss_graphic[:-1]):
                np.save('./result/BRITS_data.npy', imputation)
                print("")
                print(str(n) + "Epoch is saved! val_loss: {:0.3f}".format(val_loss_graphic[-1]))
        progress.set_description("Epoch {} loss: {:0.4f}, val_loss: {:0.4f}".format(n, epoch_loss, val_loss_graphic[-1]))

    def fuc_(x):
        for i in range(1, len(att_col)):
            if pd.isna(x[att_col[i]]):
                index_ = x.name
                x[att_col[i]] = result[index_][i - 1]
        return x

    df_ = df.apply(fuc_, axis=1)
    df_.to_csv("./result/imputed_data.csv")
    history = {"loss": loss_graphic, "val_loss": val_loss_graphic, "imputation": imputation}
    return history


if __name__ == "__main__":
    # Basic Setting
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Hyper-parameters
    epoch = 2000
    learning_rate = 0.01
    batch_size = 128
    hidden_size = 64
    num_workers = 4
    # Set weight for Imputation (input a real number between 0 and 1)
    impute_weight = 1
    # Set weights for classification and regression (enter a real number between 0 and 1)
    # It is only set to 0 in case of Imputation.
    label_weight = 0
    file_name = "train_data.csv"
    # set learning units
    separate_unit = 8760

    # Mainstream Start
    # Set json from raw data
    util_data = util.DataPreprocess(separate_unit, file_name)
    # return the required parameters from this
    seq_len, att_len = util_data.need_parameters()
    # Split data to fit a given batch-size
    data_iter = util.get_loader(batch_size=batch_size, num_workers=num_workers)
    print("Start Train!")
    # Call the model
    model = brits.Model(hidden_size, impute_weight, label_weight, seq_len, att_len).cuda()
    # Call the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # train
    history = model_train(model, optimizer, data_iter, file_name)

    # result
    with open("./json/loss_data.json", "w") as f:
        value = json.dumps(history["loss"])
        f.write(value)
    with open("./json/val_loss_data.json", "w") as f:
        value = json.dumps(history["val_loss"])
        f.write(value)
    plt.figure(figsize=(5, 5))
    plt.plot(history["loss"], label="train_loss", color="blue")
    plt.show()
    plt.figure(figsize=(5, 5))
    plt.plot(history["val_loss"], label="validation_loss", color="red")
    plt.show()