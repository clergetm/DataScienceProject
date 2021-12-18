import glob
import os

import numpy as np
import pandas as pd


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
