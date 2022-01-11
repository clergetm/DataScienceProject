import math
import pickle

import numpy as np
import pandas as pd

from Test.Test import test
from Train.Train import train
from Utils.TreeClassUtils.mainDecisionTreeClassifier import decision_tree_classifier
from Utils.Utils import delete_files

pickle_folder = "Files/Out/Pickles/TreeClass/"


def train_test_multiple(time_window_length, non_overlapping_length, pickle_name, percentage):
	"""
	The train_test_multiple study
	:param int time_window_length: The number of date in one window
	:param int non_overlapping_length: The number of different elements between two window following each other
	:param str pickle_name: Name of the .pickle file
	:param int percentage: The percentage of train_x values against test_x values (percentage vs (100-percentage))
	"""
	
	print("Starting train_test_multiple Protocol")
	train_x, train_y = train(time_window_length, non_overlapping_length)
	test_x, test_y = test(time_window_length, non_overlapping_length)
	
	# Get the number of instance equal to the percentage
	n = math.ceil(percentage * test_x.shape[0] / (100 - percentage))
	
	# Change the number of element for the training set
	train_x = train_x.sample(n)
	train_y = np.array(train_y)[train_x.index.values]
	
	# Save the dataset as Pickle file
	pickle_file = pickle_folder + pickle_name + ".pickle"
	with open(pickle_file, "wb") as f:
		# X_train, X_test, y_train, y_test
		pickle.dump([train_x, test_x, train_y, test_y], f)
	
	# Run the decision tree classifier
	decision_tree_classifier(dataset_path_name=pickle_file)
	# Create a separation between each protocol
	print("*" * 100)


def k_cross_validation(time_window_length, non_overlapping_length, k, pickle_name):
	"""
	NOT WORKING
	The train_test_multiple study
	:param int time_window_length: The number of date in one window
	:param int non_overlapping_length: The number of different elements between two window following each other
	:param int k : The number of fold
	:param str pickle_name: Name of the .pickle file
	"""
	print("NOT WORKING")
	return None
	print(f"Starting {k}_cross_validation Protocol")
	train_x, train_y = train(time_window_length, non_overlapping_length)
	test_x, test_y = test(time_window_length, non_overlapping_length)
	
	# Get all the data as one variable
	total_x = pd.concat([train_x, test_x])
	total_y = train_y + test_y
	
	# n the number of instances that we need to take for each fold
	n = math.ceil(total_x.shape[0] / k)
	
	new_train_x = pd.DataFrame()
	new_train_y = []
	
	# Get the training data
	# Get k - 1 part of the data
	for i in range(k - 1):
		
		## NOT WORKING PART
		# Pick up sample
		tmp_x = total_x.sample(n, ignore_index=True)
		tmp_y = np.array(total_y)[tmp_x.index.values]
		
		# Remove the sample from the total lists
		total_x = total_x[~total_x.isin(tmp_x)]
		total_y = [j for i, j in enumerate(total_y) if i not in tmp_y]
		## PART END HERE
		
		# Save the picked data
		new_train_x = pd.concat([new_train_x, tmp_x], ignore_index=True)
		new_train_y = new_train_y + tmp_y.tolist()
	
	# Get the last part of the data
	new_test_x = total_x.sample(n)
	new_test_y = np.array(total_y)[new_test_x.index.values]
	
	# Save the dataset as Pickle file
	pickle_file = pickle_folder + pickle_name + ".pickle"
	with open(pickle_file, "wb") as f:
		# X_train, X_test, y_train, y_test
		pickle.dump([new_train_x, new_test_x, new_train_y, new_test_y], f)
	
	# Run the decision tree classifier
	decision_tree_classifier(dataset_path_name=pickle_file)
	# Create a separation between each protocol
	print("*" * 100)


if __name__ == "__main__":
	delete_files(pickle_folder)
	# # 40/60
	train_test_multiple(200, 10, "1__train_test_200_10", 40)
	# train_test_multiple(200, 20, "1__train_test_200_20", 40)
	# train_test_multiple(100, 5, "1__train_test_100_5", 40)
	# #
	# # 50/50
	# train_test_multiple(200, 10, "2__train_test_200_10", 50)
	# train_test_multiple(200, 20, "2__train_test_200_20", 50)
	# train_test_multiple(100, 5, "2__train_test_100_5", 50)
	#
	# # 60/40
	# train_test_multiple(200, 10, "3__train_test_200_10", 60)
	# train_test_multiple(200, 20, "3__train_test_200_20", 60)
	# train_test_multiple(100, 5, "3__train_test_100_5", 60)
	#
	# # 70/30
	# train_test_multiple(200, 10, "4__train_test_200_10", 70)
	# train_test_multiple(200, 20, "4__train_test_200_20", 70)
	# train_test_multiple(100, 5, "4__train_test_100_5", 70)

	# # 10 cross validation
	# NOT WORKING
	# k_cross_validation(200, 10, 10, "5__10_cross_200_10")
	# k_cross_validation(200, 20, 10, "5__10_cross_200_20")
	# k_cross_validation(100, 5, 10, "5__10_cross_100_5")
