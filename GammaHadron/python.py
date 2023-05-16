import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


cols = ["fLength", "fWidth", "fSize", "fConc",  "fConcl", "fAsym", "fM3Long", "fM3Trans","fAlpha", "fDist", "class"]

df = pd.read_csv("magic04.data", names = cols)
df["class"] = (df["class"] == "g").astype(int)

print(df)

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

print(len(train[train["class"]==1])) #gamma
print(len(train[train["class"]==0]))

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y


train, X_train, y_train = scale_dataset(train, oversample = True)

print(len(y_train))
print(sum(y_train == 1))
print(sum(y_train == 0))
