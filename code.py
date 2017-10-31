import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

fn = r'C:\Users\DELL I5558\Desktop\Python\ELEC5222\kNN\NSW-ER01-8.csv'
dataframe = pd.read_csv(fn)
dataset = dataframe.values
X = dataset[:, 0:23].astype(float)
Y = dataset[:, 23]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=25)
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
knn_scaled = pipeline.fit(X_train, y_train)
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

print('Accuracy with Scalling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scalling: {}'.format(knn_unscaled.score(X_test, y_test)))
