import time
import glob

import numpy as np
import pandas as pd
from scipy.stats import zscore  # Normalize DataFrame

from Utils.Get import get_data, get_duplicate, get_nan, get_specs
from Utils.Protocol import protocol

########################################################################################################################
#                                                  USER PARAMETERS                                                     #
########################################################################################################################

# Path to the training dataset from the Train/main.py file
path = "../Files/Docs/MHEALTHDATASET/TrainSet/"

# List of files used as training set
files = [f"mHealth_subject{str(i)}.txt" for i in range(1, 2)]

if __name__ == '__main__':
	start_time = time.time()
	for file in files:
		# Create the dataFrame
		dataset_path = path + file
		df = get_data(dataset_path, sep="\\\t")
		
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
		
		activities = []
		# Get each possible activity
		activities_id = np.unique(df['Label'])
		
		# activity_min will be useful for flattening all the activity at the same amount of instances
		activity_min = df.shape[0]  # shape[0] to initialize at a high value
		
		for id_value in activities_id:
			activity = df[df['Label'] == id_value]
			instances = activity.shape[0]  # Get the number of instances
			
			# Change the min value
			activity_min = instances if instances < activity_min else activity_min
			
			# print(f"{id_value}: {instances}")
			activities.append(activity)
		
		# Now that we have the activities and a minimum : we need to flatten them
		flatten_activities = []

		for activity in activities:
			if activity.shape[0] > activity_min:
				
				#  Get the minimum number of instances chosen at random
				activity = activity.sample(activity_min)
				
				# Normalizes DataFrame using the z score method
				activity.apply(zscore)
				
			activity.reset_index(inplace=True, drop=True)  # Reset the index and not keep the previous one
			flatten_activities.append(activity)
		
		print("X" * 100)
	
		# After the cleaning and the activities' creation, we can launch the protocol
		protocol(flatten_activities, time_window_length=0, non_overlapping_length=0,
		         # split to remove the extension
		         pickle_file=file.split(sep='.')[0], folder=file.split(sep='.')[0])
	
	end_time = time.time()
	print(f"total time = {end_time - start_time}")
