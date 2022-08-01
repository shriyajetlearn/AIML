import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("car.data")
data.columns = ("sales", "maintainence", "doors","persons", "boot_space", "safety", "class")

print(data.head())
print(data.describe())
print(data.info())

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

data["sales"] = label_encoder.fit_transform(data["sales"])
data["maintainence"] = label_encoder.fit_transform(data["maintainence"])
data["boot_space"] = label_encoder.fit_transform(data["boot_space"])
data["safety"] = label_encoder.fit_transform(data["safety"])
data["doors"] = label_encoder.fit_transform(data["doors"])
data["persons"] = label_encoder.fit_transform(data["persons"])
data["class"] = label_encoder.fit_transform(data["class"])

X = data[["sales", "maintainence", "doors","persons", "boot_space", "safety"]]
Y = data["class"]

from skklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)

from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(X_train, Y_train)  

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(Y_test, y_pred)
sns.heatmap(matrix, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(Y_test, y_pred))
