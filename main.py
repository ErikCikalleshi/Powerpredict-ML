import pandas as pd
import os
import sklearn

DATASET_PATH = "."


powerpredict = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv"))
dummies = pd.get_dummies(powerpredict)
print(dummies.corr())





