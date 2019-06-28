import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

import tensorflow as tf

#Read the csv dataframe
dataframe = pd.read_csv("/home/senthil/projects/tensorflow/Tensorflow-Bootcamp-master/02-TensorFlow-Basics/cal_housing_clean.csv")
print(dataframe.head())

# list of dtypes to include
include = ['object', 'float', 'int']
# dataframe description
desc = dataframe.describe(include=include)
# description transposed
desc.transpose()
print(desc.transpose())

# Split using sklearn
## 70% train and 30% test
x_data = dataframe.drop('medianHouseValue', axis=1)
labels = dataframe['medianHouseValue']
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.30, random_state=101)

#Optional
##describe the train data transposed
desc = X_train.describe(include=include)
desc.transpose()
print(desc.transpose())

#Optional
##describe the test data transposed
desc = X_test.describe(include=include)
desc.transpose()
print(desc.transpose())

# use MinMaxScaler to preprocess for both train and test\
scaler = MinMaxScaler()
scaler.fit_transform(X_train)

X_train_scaled = pd.DataFrame(data=X_train, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(data=scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

# Create the feature columns
housingMedianAge = tf.feature_column.numeric_column('housingMedianAge')
totalRooms = tf.feature_column.numeric_column('totalRooms')
totalBedrooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
medianIncome = tf.feature_column.numeric_column('medianIncome')

feat_cols = [housingMedianAge, totalRooms, totalBedrooms, population, households, medianIncome]

tf.logging.set_verbosity(tf.logging.INFO)

#Create input function
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train_scaled, y = y_train, batch_size=10, num_epochs=20000, shuffle=True)


#Create the estimator model WITH DNN
dnn_model = tf.estimator.DNNRegressor(hidden_units=[6,6,6,6,6], feature_columns=feat_cols)
dnn_model.train(input_fn=input_func, steps=10000)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
#prediction generator
pred_gen = dnn_model.predict(pred_input_func)
#as list
predictions = list(pred_gen)

prediction_vals = []
for pred in predictions:
    print(pred['predictions'])
    prediction_vals.append(pred['predictions'])

print(prediction_vals)

rmse = mean_squared_error(y_test, prediction_vals)**0.5
print(rmse)


#Create the estimator model WITH =LNN
# lnn_model = tf.estimator.LinearRegressor(feature_columns=feat_cols)
# lnn_model.train(input_fn=input_func, steps=10000)
#
# lin_pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
# #prediction generator
# lin_pred_gen = lnn_model.predict(lin_pred_input_func)
# #as list
# lin_predictions = list(lin_pred_gen)
#
# lin_prediction_vals = []
# for pred in lin_predictions:
#     print(pred['predictions'])
#     lin_prediction_vals.append(pred['predictions'])
#
# print(lin_prediction_vals)
#
# lin_rmse = mean_squared_error(y_test, lin_prediction_vals)**0.5
# print(lin_rmse)

