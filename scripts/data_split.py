import pandas as pd
import numpy as np
np.random.seed(2020)
df_raw = pd.read_csv("../dataset/train.csv", encoding="utf8")
sample_index = np.arange(len(df_raw))
np.random.shuffle(sample_index)


images, labels = df_raw.values[:, 0][sample_index], df_raw.values[:, 1][sample_index]
data_size = len(df_raw)


train_size = int(data_size * 0.8)

train_images, valid_images = images[:train_size], images[train_size:]
train_labels, valid_labels = labels[:train_size], labels[train_size:]

pd.DataFrame({'id': train_images, 'label': train_labels}).to_csv("../dataset/new_train.csv", index=False)
pd.DataFrame({'id': valid_images, 'label': valid_labels}).to_csv("../dataset/new_valid.csv", index=False)