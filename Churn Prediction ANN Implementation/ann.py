# -*- coding: utf-8 -*-

# Artificial Neural Network

# Data Preprocessing

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encode categorical data (Label encode then do one hot encoding)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create ANN Classifier
import keras 
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense 
from keras.layers import Dropout

# Initialise the ANN
classifier = Sequential()

# Add input layer and first hidden layer
classifier.add(Dense(units = 6, activation = 'relu',
                     kernel_initializer = 'uniform',
                     input_dim = 11))
classifier.add(Dropout(rate = 0.1))
# Add second hidden layer
classifier.add(Dense(units = 6, activation = 'relu',
                     kernel_initializer = 'uniform'))
classifier.add(Dropout(rate = 0.1))
# Add output layer
classifier.add(Dense(units = 1, activation = 'sigmoid',
                     kernel_initializer = 'uniform'))

#Compile ANN (Apply stochastic gradient descent on the whole network)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

#Fit ANN to the training set
classifier.fit(x=X_train, y=y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

y_test = (y_test > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Use the NN to predict for one customer
customer = np.array([0,0,600,1,40,3,60000,2,1,1,50000]).reshape(1,-1)
customer = sc.transform(customer)
customer_pred = classifier.predict(customer)
#7.9% probability of leaving = False, customer wont leave


# K-fold CV implementation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier(optimizer):
    classifier = Sequential()

    classifier.add(Dense(units = 6, activation = 'relu',
                     kernel_initializer = 'uniform',
                     input_dim = 11))

    classifier.add(Dense(units = 6, activation = 'relu',
                     kernel_initializer = 'uniform'))

    classifier.add(Dense(units = 1, activation = 'sigmoid',
                     kernel_initializer = 'uniform'))

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy',
                       metrics = ['accuracy'])

    return classifier

#build a new classifier trained with k-fold CV
classifier = KerasClassifier(build_fn = build_classifier, 
                             batch_size = 10, 
                             epochs = 100)

accuracies = cross_val_score(classifier,X_train, y_train, cv = 10, n_jobs = 1)
cv_score = accuracies.mean()
variance = accuracies.std()

# If you have high variance in the accuracies, apply dropout 

# Parameter Tuning (create dict with keys = params, values = all diff values)
parameters = {'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
# fit gridsearch ANN to the training set
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#best param {'batch_size': 32, 'epochs': 500, 'optimizer': 'rmsprop'}
#best acc: 85%



