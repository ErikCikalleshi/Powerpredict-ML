import pandas as pd
import os

DATASET_PATH = "."


powerpredict = pd.read_csv(os.path.join(DATASET_PATH, "powerpredict.csv"))
print(powerpredict.shape)





