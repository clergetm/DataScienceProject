import time
from math import ceil, floor

import numpy as np
import pandas as pd
from sklearn import preprocessing

from Utils.Get import get_data, get_duplicate, get_nan

########################################################################################################################
#                                                    USER PARAMETERS                                                   #
########################################################################################################################

# Path to the training dataset from the Train/main.py file
path = "../Files/Docs/MHEALTHDATASET/TrainSet/"

# List of files used as training set
files = [f"mHealth_subject{str(i)}.txt" for i in range(1, 2)]


		################################################################################################################
		#                                          DATA CLEANING                                                       #
		################################################################################################################
		
		# print("Get information about duplicates and nan in the dataFrame")
		# print("Nan: ")
		# print(get_nan(df))  # No Nan Values
		# print("Duplicates: ")
		# print(get_duplicate(df))  # No Duplicates Values
		
		# Creating Classes
		# The Label '0' need to be deleted : it corresponds to no class
		df.drop(index=df[df['Label'] == 0].index, axis='index', inplace=True)
		
		classes = []
		# Get each possible class
		class_id = np.unique(df['Label'])
		
		# If a class have fewer instances than the mean then the class is deleted
		mean = floor(np.mean([df[df['Label'] == id_value].shape[0] for id_value in class_id]))
		
		# class_min will be useful for flattening all the class at the same amount of instances
		class_min = df.shape[0]
		
		for id_value in class_id:
			_class = df[df['Label'] == id_value]
			instances = _class.shape[0]  # Get the number of instances
			
			# Validation
			if instances < mean:
				df.drop(index=_class.index, axis='index', inplace=True)
			else:
				# Change the min value
				class_min = instances if instances < class_min else class_min
				# print(f"{id_value}: {instances}")
				classes.append(_class)
		
		# Now that we have the classes and the minimum : we need to flatten the classes
		flatten_classes = []
		for _class in classes:
			if _class.shape[0] > class_min:
				#  Get the minimum number of instances chosen at random
				_class = _class.sample(class_min)
			
			_class.reset_index(inplace=True)
			flatten_classes.append(_class)
		
		print(flatten_classes)
		print("X" * 100)

	end_time = time.time()
	print(f"total time = {end_time - start_time}")