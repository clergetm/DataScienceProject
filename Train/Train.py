import time
from typing import Tuple

import numpy as np
import pandas as pd

from Utils.Get import get_data  # Getter pd.DataFrame
from Utils.Graphics import plot_stats  # Plotting the stats of features
from Utils.Protocol import protocol  # The protocol of the project

from Utils.Utils import normalize  # To normalize DataFrame
from Utils.Utils import to_flatten_df  # To flatten DataFrame


def train(time_window_length, non_overlapping_length, with_plot_stats=False) -> Tuple[pd.DataFrame, list]:
	"""
	Main function for the training set
	:param int time_window_length: The number of date in one window
	:param int non_overlapping_length: The number of different elements between two window following each other
	:param bool with_plot_stats: True if the user want to create the plots of the stats.
		May cause drastic slowdown performance
	:returns: x_data and y_data, x is the complete dataset and y the list of the class of each x's instance
	:rtype (new_x_data, new_y_data): (pd.DataFrame, list)
	"""
	####################################################################################################################
	#                                                  USER PARAMETERS                                                 #
	####################################################################################################################
	
	# Path to the training dataset from the Train/Train.py file
	path = "../Files/Docs/MHEALTHDATASET/TrainSet/"
	
	# List of files used as training set
	subjects_files = [f"mHealth_subject{str(i)}" for i in range(1, 10)]
	subjects_list_df = []
	
	# Max number of instances for each dataset (will be changed below)
	# ceil_instances will be useful for flattening all the activity at the same amount of instances
	# So to flatten all subject's dataset
	ceil_instances = 99999999999999
	
	# Variables used in the protocol
	pickle_file = "TrainSet"
	folder_type = "Train"
	
	print("Starting main")
	start_time = time.time()
	for subject in subjects_files:
		
		# Create the dataFrame
		dataset_path = path + subject + ".txt"
		df = get_data(dataset_path, sep="\\\t", txt=False)
		################################################################################################################
		#                                          DATA CLEANING                                                       #
		################################################################################################################
		
		# Creating Classes
		# The Label '0' need to be deleted : it corresponds to no class
		df.drop(index=df[df['Label'] == 0].index, axis='index', inplace=True)
		
		activities = []
		# Get each possible activity
		activities_id = np.unique(df['Label'])
		
		for id_value in activities_id:
			activity = df[df['Label'] == id_value]
			instances = activity.shape[0]  # Get the number of instances
			
			# Change the max value
			ceil_instances = instances if instances < ceil_instances else ceil_instances
			
			# print(f"{id_value}: {instances}")
			activities.append(activity)
		
		# Create a Dataframe that refers to the subject
		df_subject = pd.concat(activities)
		df_subject["ID"] = subject  # Create a column to indicate the ID of the subject
		
		# To Flatten the DataFrame
		df_subject = to_flatten_df(to_flat=df_subject, ceil=ceil_instances)
		
		# To normalize the DataFrame
		df_subject = normalize(df_subject, minus_columns=2)  # Don't normalize 'Label' and 'ID'
		
		# Add this subject to the list of subjects_files
		subjects_list_df.append(df_subject)
	
	# Concat all subjects_files in one dataset
	df_subjects = pd.concat(subjects_list_df, ignore_index=True)
	
	####################################################################################################################
	#                                  PROTOCOLS                                                                       #
	####################################################################################################################
	
	# After the cleaning and the activities' creation, we can launch the protocol
	
	# Get the stats
	if with_plot_stats:
		print("Start plot Stats")
		features = df_subjects.columns[:-2]  # Get the columns except 'ID' and 'Label'
		plot_stats(subjects_list_df, features=features, save_path="../Files/Out/Stats/Train/")
	
	print("Launch Protocol")
	x_data, y_data = protocol(subjects_list_df, time_window_length=time_window_length,
	                          non_overlapping_length=non_overlapping_length,
	                          pickle_file=pickle_file, folder_type=folder_type)
	
	end_time = time.time()
	print(f"total time = {end_time - start_time}")
	return x_data, y_data
