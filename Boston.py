'''DNNRegressor with custom input_fn for Housing dataset'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

#define the column names for the data set
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]

FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]

LABELS = "medv"

#create an input function, which will accept a pandas Dataframe 
#and return feature column and label values as Tensors
def input_fn(data_set):
	feature_cols = {k: tf.constant(data_set[k].values)
				for k in FEATURES}
	labels = tf.constant(data_set[LABELS].values)
	return feature_cols, labels

def main(unused_argv):
	#load data
	training_set = pd.read_csv("boston_train.csv", skipinitialspace=True, 
								skiprows=1, names=COLUMNS)

	test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
							skiprows=1, names=COLUMNS)

	prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
							skiprows=1, names=COLUMNS)

	#create a list of FeatureColumns used for training for the input data
	feature_cols = [tf.contrib.layers.real_valued_column(k)
					for k in FEATURES]

	#instantiate a DNNRegressor
	regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
												hidden_units=[10, 10],
												model_dir="d:/Workspace/tensorflow/boston_model")

	#training the regressor
	regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

	#evaluating the model
	ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
	loss_score = ev["loss"]
	print("Loss: {0:f}".format(loss_score))

	#making predictions
	y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
	# .predict() returns an iterator; convert to a list and print predictions
	predictions = list(itertools.islice(y, 6))
	print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
	tf.app.run()