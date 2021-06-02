import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#get data and make dataframe
df = pd.read_csv("Data/Iris data")

#build model
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS', axis=1))

#transform data
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis =1))

df.columns

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

#build model
from sklearn.model_selection import train_test_split
X = scaled_features
y= df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred=knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

error_rate = []

for i in range(1,40):
    knn =KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test)) #mean is taken if it is not equal to prediction

from matplotlib import figure

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed', marker='o', markerfacecolor='red',markersize=10)
plt.title('error rate vs K values')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn =KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

##should work, but doesn't