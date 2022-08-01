"""
Notes - 

vectors - quantities with direction, displacement, velocity, acceleration, force
scaler - quanties without directions speed, distance, work, energy, power


support vector machines
--supervised learning algorithm
--highly effiecient
--data is separated by a hyperplane
-- different functions available make it possible to use in variety of situations

"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd

cancer_dict = datasets.load_breast_cancer()
print(cancer_dict.keys())
cancer_data = pd.DataFrame(cancer_dict.data)
cancer_data.columns = cancer_dict.feature_names
cancer_data["isCancer"] = cancer_dict.target

print(cancer_data.info())
print(cancer_data.head())

Y = cancer_data["isCancer"]
cancer_data.drop("isCancer", axis = 1)
X = cancer_data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 50)

cls = svm.SVC(kernel = "linear")
cls.fit(X_train, Y_train)
y_pred = cls.predict(X_test)

print("Accuracy Score -", metrics.accuracy_score(Y_test, y_pred))
