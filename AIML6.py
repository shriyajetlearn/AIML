import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("student-mat.csv")

print(data.head())
print(data.describe())
print(data.info())

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data["school"] = label_encoder.fit_transform(data["school"])
data["sex"] = label_encoder.fit_transform(data["sex"])
data["famsize"] = label_encoder.fit_transform(data["famsize"])
data["Pstatus"] = label_encoder.fit_transform(data["Pstatus"])
data["Mjob"] = label_encoder.fit_transform(data["Mjob"])
data["Fjob"] = label_encoder.fit_transform(data["Fjob"])
data["reason"] = label_encoder.fit_transform(data["reason"])
data["guardian"] = label_encoder.fit_transform(data["guardian"])
data["schoolsup"] = label_encoder.fit_transform(data["schoolsup"])
data["famsup"] = label_encoder.fit_transform(data["famsup"])
data["paid"] = label_encoder.fit_transform(data["paid"])
data["activities"] = label_encoder.fit_transform(data["activities"])
data["nursery"] = label_encoder.fit_transform(data["nursery"])
data["higher"] = label_encoder.fit_transform(data["higher"])
data["internet"] = label_encoder.fit_transform(data["internet"])
data["romantic"] = label_encoder.fit_transform(data["romantic"])


data.drop('G1', axis = 1, inplace = True)
data.drop('G2', axis = 1, inplace = True)

X = data[["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]]

Y = data["G3"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 5)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(Y_test, y_pred)
sns.heatmap(matrix, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(Y_test, y_pred))
