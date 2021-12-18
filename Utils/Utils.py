import glob
import os

import numpy as np
import pandas as pd

from Utils.Get import get_data  # Getter pd.DataFrame


def data_cleaning(folder, subjects_names) -> list:
	"""
	Create and clean the datasets
	:param str folder: the path to the folder where the dataset files are saved
	:param list subjects_names: the list of the files names
	:return: a list that contain the dataset of each file
	:rtype subjects_df: list
	"""
	####################################################################################################################
	#                                             USER PARAMETERS                                                      #
	####################################################################################################################

	# result list that will be returned
	subjects_df = []
	
	# Max number of instances for each dataset (will be changed below)
	# ceil_instances will be useful for flattening all the activity at the same amount of instances
	# So to flatten all subject's dataset
	ceil_instances = 99999999999999
	
	####################################################################################################################
	#                                              DATA CLEANING                                                       #
	####################################################################################################################
	
	for subject in subjects_names:
		
		# Create the dataFrame
		dataset_path = folder + subject + ".txt"
		df = get_data(dataset_path, sep="\\\t", txt=False)
		
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

			# Add the activity to the list
			activities.append(activity)
		
		# Create a Dataframe that refers to the subject
		df_subject = pd.concat(activities)
		df_subject["ID"] = subject  # Create a column to indicate the ID of the subject
		
		# To Flatten the DataFrame
		df_subject = to_flatten_df(to_flat=df_subject, ceil=ceil_instances)
		
		# To normalize the DataFrame
		df_subject = normalize(df_subject, minus_columns=2)  # Don't normalize 'Label' and 'ID'
		
		# Add this subject to the list of subjects' list
		subjects_df.append(df_subject)
	
	return subjects_df
	
	
def delete_files(path):
	"""
	Delete files inside the given folder
	:param str path: the path to a folder to clear
	"""
	if os.path.isdir(path):
		for f in glob.glob(path + "/*"):
			os.remove(f)
			

def to_flatten_df(to_flat, ceil) -> pd.DataFrame:
	"""
	To Flatten the DataFrame at the same ceil for each activity
	:param pd.DataFrame to_flat: the DataFrame to flatten
	:param ceil: the ceil for each activity
	:return: The flattened DataFrame
	:rtype: pd.DataFrame
	"""
	# List that will contain flattened activities
	flattened_activities = []
	
	# Loop for each activity
	for id_activity in np.unique(to_flat['Label']):
		tmp_activity = to_flat[to_flat['Label'] == id_activity]
		
		# Compare the length of the activity to the ceil value
		if tmp_activity.shape[0] > ceil:

			#  Get the minimum number of instances chosen at random
			tmp_activity = tmp_activity.sample(ceil)
				
		flattened_activities.append(tmp_activity)
	# return a Flattened DataFrame
	return pd.concat(flattened_activities, ignore_index=True)


def normalize(df, minus_columns=-100) -> pd.DataFrame:
	"""
	Normalize the given DataFrame
	:param pd.DataFrame df: The DataFrame to Normalize
	:param int minus_columns: The number of column to NOT Normalize starting from the rightmost column
		The default value of -100 correspond to removing 0 column
	:return result: the normalized DataFrame
	:rtype result: pd.DataFrame
	
	see also: From https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
	"""
	
	# Create a copy of the DataFrame to work with
	result = df.copy()
	
	for feature_name in df.columns[:-minus_columns]:
		# Get minimum and maximum values of the feature
		min_value = df[feature_name].min()
		max_value = df[feature_name].max()
		
		# Normalize the feature
		result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
		
	# Return the normalized DataFrame
	return result
