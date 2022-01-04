#import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MyLogReg import MyLogisticRegression as mlr


#import data and some preprocessing
df = pd.read_csv(r'/Users/ismail/Desktop/LogReg/heart.csv')
df = df.rename(columns={"cp": "chest_pain_type","trestbps":"resting_blood_pressure","chol":"serum_cholestoral","fbs":"fasting_blood_sugar","restecg":"resting_ecg","thalach":"maximum_heart_rate","exang":"exercise_angina","oldpeak":"st_depression","ca":"major_colored","slope":"st_slope","thal":"thalassemia"})

y = df.target.values
X = df.drop(["target","sex","chest_pain_type","fasting_blood_sugar","resting_ecg","exercise_angina","st_slope","thalassemia"],1)

# Split data %80 for training and %20 for test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

#normalize the data
X_train = (X_train - np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test = (X_test - np.min(X_test))/(np.max(X_test)-np.min(X_test)).values


x_train = X_train.T
y_train = y_train.T
x_test = X_test.T
y_test = y_test.T


# create a model from our custom logistic regression model class
logreg = mlr()
# train the model with training data
parameters, gradients, cost_list = logreg.fit(x_train, y_train, 0.3, 1000)
# test the model
logreg.predict(parameters["weight"],parameters["bias"],x_test)


