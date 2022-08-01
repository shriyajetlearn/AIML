# Topic - Linear Regression

# Revise the kid about following concepts - 
# Machine Learning
# Regression

# Explain What we are trying to achieve by Linear Regression ?
# Explain basics about Linear Equations - y = mx + c, m = slope, c = intercept
# For drawing a unique line you need m and c

# Talk about Best-Fit and Mean-Sqaured Error, Why normal way of calculating errors will not work. What do we acheive by Mean-Sqaured Error.

# Refer - https://machinelearningmastery.com/simple-linear-regression-tutorial-for-machine-learning/

#m = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#c = mean(y) – m * mean(x)

# Explain by taking basic x and y values given in the tutorial how we calculate the above values from the dataset.
# Don't go into the explanation about how we arrive at above formulas for m and c

# Repeat the same thing using code


def findMean(x):
  return sum(x)/len(x)

x = [1,2,4,3,5]
y = [1,3,3,2,5]

meanX = findMean(x)
meanY = findMean(y)
num = 0
den = 0
for i in range(len(x)):
  num = num + ((x[i] - meanX) * (y[i] - meanY))
  den = den +  pow((x[i] - meanX), 2)

m = num / den
print("m =", m)
c = round(meanY - m * meanX,1)
print("c =", c)

# Show how Libraries could be used. Give them heads-up that following only prebuilt implementations would be used because of its ease.
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([[1],[2],[4],[3],[5]])
y = np.array([[1],[3],[3],[2],[5]])

reg = LinearRegression().fit(X, y)

print("m =", reg.coef_)
print("c =", reg.intercept_)