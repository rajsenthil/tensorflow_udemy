import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf

dataframe = pd.read_csv('/home/senthil/projects/tensorflow/Tensorflow-Bootcamp-master/02-TensorFlow-Basics/census_data.csv')

print(dataframe.head())

#x_data = dataframe.drop('income_bracket', axis=1)
y_data = dataframe['income_bracket']

#print(x_data.head())
print(y_data.drop_duplicates())

mapping = {' <=50K': 0, ' >50K': 1}
# one of applying 0 OR 1 for <= or > 50 K
# X_data = dataframe.replace({'income_bracket': mapping})
# print(X_data.head())

#Using pandas apply function to assign 0 OR 1 for <= 50K or > 50K
dataframe['income_bracket_bool'] = dataframe.apply(lambda x: 0 if ' <=50K' in x['income_bracket'] else 1, axis=1)
X_data = dataframe.drop('income_bracket', axis=1)

y_data = dataframe['income_bracket_bool']

print(X_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30, random_state=101)

print(dataframe.columns)

tf.logging.set_verbosity(tf.logging.INFO)

age            = tf.feature_column.numeric_column('age')
education_num  = tf.feature_column.numeric_column('education_num')
capital_gain   = tf.feature_column.numeric_column('capital_gain')
capital_loss   = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

workclass      = tf.feature_column.categorical_column_with_hash_bucket('workclass', hash_bucket_size=1000)
education      = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status', hash_bucket_size=1000)
occupation     = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
relationship   = tf.feature_column.categorical_column_with_hash_bucket('relationship', hash_bucket_size=1000)
race           = tf.feature_column.categorical_column_with_hash_bucket('race', hash_bucket_size=100)
gender         = tf.feature_column.categorical_column_with_vocabulary_list('gender', ['Female', 'Male'])
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country', hash_bucket_size=200)

# income_bracket = tf.feature_column.numeric_column('income_bracket_bool')

assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=2)
feat_cols = [workclass, education, education_num, marital_status, occupation, relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country]


# Input function for training
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size=100, num_epochs=1000, shuffle=True)
#Model for training
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)
model.train(input_fn=input_func, steps=1000)

# Input function for prediction
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
# All these predictions will be feed dictionary
my_pred = list(predictions)

final_preds = [pred['class_ids'][0] for pred in my_pred]
print(final_preds)

report = classification_report(y_true=y_test, y_pred=final_preds)
print(report)