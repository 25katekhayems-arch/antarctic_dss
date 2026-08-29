import numpy as np
import torch
import torch.nn as nn
from model import SeaIceForecaster

SEQ_LEN = 10
FORECAST_LEN = 5
BATCH_SIZE = 8
EPOCHS = 15
DEVICE = "cpu"  # numpy-only version, no GPU needed


def load_data():
    data = np.load("sic_data.npy")
    return data


def make_windows(data, seq_len, forecast_len):
    n_days = data.shape[0]
    X, Y = [], []
    for t in range(n_days - seq_len - forecast_len + 1):
        X.append(data[t : t + seq_len])
        Y.append(data[t + seq_len : t + seq_len + forecast_len])
    X = np.stack(X)[:, :, None, :, :]
    Y = np.stack(Y)[:, :, None, :, :]
    return X.astype(np.float32), Y.astype(np.float32)


if __name__ == "__main__":
    print("Training is optional - the numpy forecaster works without it.")
    print("Run 'python region.py' to verify data loading works.")
