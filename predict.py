import pandas as pd
import os
import pickle
data = pd.read_csv('Data/Iris data Predict')

#Shows the first 5 rows of the data
data.head()

y_variable = data['TARGET CLASS']

X= data.loc[:, data.columns != 'TARGET CLASS']
X = X.loc[:, X.columns != 'NUMBER']

with open('classifier_decision_tree.pkl', 'rb') as f:
    loaded_classifier = pickle.load(f)
y_predDT = loaded_classifier.predict(X)
print(' ')
print('----------------------')
print('Decision Tree',y_predDT)
print('----------------------')
print(' ')

