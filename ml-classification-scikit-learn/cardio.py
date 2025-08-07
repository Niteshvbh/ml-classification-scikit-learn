import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("E:\Projects\Python\Coursera\Beginner\Scikit-Learn For Machine Learning Classification Problems\cardio.csv", encoding = "latin1", sep = ";")
print(df.head())

df = df.drop(columns = "id")

df["age"] = df["age"]/365

df.hist(bins = 30, figsize = (20, 20), color = "yellow")
plt.show()

corr_matrix = df.corr()
print(corr_matrix)

plt.figure(figsize = (16, 16))
sns.heatmap(corr_matrix, annot = True)
plt.show()

# split the dataframe into target and features
y = df["cardio"]
X = df.drop(columns =["cardio"])
print(X.shape)
print(y.shape)

# splitting the data in to test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from xgboost import XGBClassifier

xgb_classifier = XGBClassifier(objective = "binary:logistic", eval_metric = "error", learning_rate = 0.1, n_estimators = 100)
xgb_classifier.fit(X_train, y_train)

result = xgb_classifier.score(X_test, y_test)
print("Accuracy : {}".format(result))

y_predict = xgb_classifier.predict(X_test)
print(y_predict)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, fmt = "d", annot = True)









