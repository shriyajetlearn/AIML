"""
Notes
--Curse of Dimensionality
--30 features of my body

--y = mx + c # simple linear regression with one feauture
--y = m1x1 + m2x2 + m3x3 + c  # multivariable linear regression with 3 features
--y = m1x1 + m2x2 + m3x3 .....   + c # kind of tricky / diffucult

What is PCA ?
--Principle Component Analysis
--to remove inconsistencies
--redundant data
--highly-corelated features

Steps -
1) Standardize of data
2) Compute the covaraince matrix

    negative covariance = indirectly propotional
    positive covaraince = directly propotional
    0 - no corelation independent variables

3) compute the eigen values and eigen vectors
4) compute your principal components

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
raw_data = load_breast_cancer()
#print(raw_data.keys())
#print(raw_data["target"])
#print(raw_data["target_names"])
#print(raw_data["DESCR"])
#print(raw_data["feature_names"])

data = pd.DataFrame(raw_data["data"], columns = raw_data["feature_names"])
#print(data.info())
#print(data.head())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)

scaled_data = scaler.transform(data)
print(scaled_data)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(scaled_data)

new_data = pca.transform(scaled_data)
print(scaled_data.shape)
print(new_data.shape)

plt.figure(figsize = (10, 10))
plt.scatter(new_data[:, 0], new_data[:, 1], c = raw_data["target"])
plt.xlabel("First Principal Component")
plt.ylabel("Second Princiapl Component")
plt.show()
