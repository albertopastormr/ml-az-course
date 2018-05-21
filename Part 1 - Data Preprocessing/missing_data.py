import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 3]

# Fixing the missing data fields
# iloc --> Integer indexing ; loc --> tag-row,tag-column types indexing
impute = Imputer(missing_values= 'NaN', strategy='mean', axis=0)
impute = impute.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = impute.transform(X.iloc[:, 1:3])