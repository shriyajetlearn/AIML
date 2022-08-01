# Topic - KNN

# Explain What is feature scaling, How does it help ?
# Explain the min-max scaling running on an example dataset
# Explain the concept of KNN, by giving example and dry running visually

# The code to be done is below - 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data.csv")

print(data.head())

X = data[["sepal_length","sepal_width","petal_length","petal_width"]]
Y = data["species"]

print(X.head())
print(Y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 5)

from sklearn.preprocessing import StandardScaler, LabelEncoder

standard_scaler  = StandardScaler()
standard_scaler.fit_transform(X_train)

label_encoder = LabelEncoder()
label_encoder.fit_transform(Y_train)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbours = 5)
classifier.fit(X_train, Y_train)

standard_scaler.transform(X_test)
y_pred = classifier.predict(X_test)

label_encoder.transform(Y_test)

from sklearn.metrics import classification_report, confusion_matrix

matrix = confusion_matrix(Y_test, y_pred)

sns.heatmap(matrix, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(Y_test, y_pred))
