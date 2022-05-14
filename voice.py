import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


csv_file = pd.read_csv('./data/gender_voice.csv')

raw_data = csv_file.iloc[:, :20]
data = raw_data.values

raw_label = csv_file.iloc[:, 20]
labels = raw_label.values

idx = np.arange(data.shape[0])
np.random.shuffle(idx)
data = data[idx, :]
labels = labels[idx]

test_ratio = 0.2
valid_ratio = 0.1

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_ratio, random_state=1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio, random_state=1)
print(x_train.shape)
