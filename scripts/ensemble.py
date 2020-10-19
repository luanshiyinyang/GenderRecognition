import pandas as pd
from glob import glob
import numpy as np
from scipy import stats

rsts = glob("../results/*.csv")

preds = []
for file in rsts:
    df = pd.read_csv(file, encoding="utf8")
    preds.append(df.values[:, 1])

preds = np.array(preds)
final_rst = stats.mode(np.array(preds))[0][0]

submit = pd.read_csv("../dataset/test.csv", encoding="utf8")
submit['label'] = final_rst
submit.to_csv("submit.csv", encoding="utf8", index=False)