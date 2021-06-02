#Import Pandas library
import pandas as pd
#import method from sklearn to split our data into training and test data
from sklearn.model_selection import train_test_split
#import DecisionTreeClassifier from the Sklearn library
from sklearn.tree import DecisionTreeClassifier

#List with attribute names (it is optional to do this but it gives a better understanding of the data for a human reader)
#attribute_names = ['variance_wavelet_transformed_image', 'skewness_wavelet_transformed_image', 'curtosis_wavelet_transformed_image', 'entropy_image', 'class']

#Read csv-file
data = pd.read_csv('data\Iris data')

#Shuffle data
data = data.sample(frac=1)

#Shows the first 5 rows of the data
data.head()

#'class'-column
y_variable = data['TARGET CLASS']

#all columns that are not the 'TARGET CLASS' or 'NUMBER'-column -> all columns that contain the attributes
X_variables = data.loc[:, data.columns != 'TARGET CLASS']
X_variables = X_variables.loc[:, X_variables.columns != 'NUMBER']

#splits into training and test data
X_train, X_test, y_train, y_test = train_test_split(X_variables, y_variable, test_size=0.2)

#Create a classifier object 
classifier = DecisionTreeClassifier() 

#Classfier builds Decision Tree with training data
classifier = classifier.fit(X_train, y_train) 

#Print that training is ready and show accuracy
print('Classifier trainiert! Akkuranz: {}%'.format(classifier.score(X_test,y_test)*100))