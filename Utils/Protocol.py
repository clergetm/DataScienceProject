import os
import pickle
from typing import Tuple

import pandas as pd

from Utils.DimUtils.dimentionalityReduction import dim_reduc_protocol
from Utils.ExtractionUtils.Extraction import get_specs


########################################################################################################################
#                                                   PROTOCOL                                                           #
########################################################################################################################


def protocol(subjects, time_window_length, non_overlapping_length, pickle_file, folder_type) -> Tuple[pd.DataFrame, list]:
	"""
	The main protocol of the Project to obtains the stats of each activity and execute the dimentionalityReduction protocol
	:param list subjects: list of pd.DataFrame
	:param int time_window_length: The number of date in one window
	:param int non_overlapping_length: The number of different elements between two window following each other
	:param str pickle_file: Name of the .pickle file
	:param str folder_type: Name of the folder
	:returns: new_x_data, the dataset generated by this function and the dimensionality reduction
	 and new_y_data the list of class of each element
	:rtype (new_x_data, new_y_data): (pd.DataFrame, list)
	"""
	
	# Variables
	subject_x_data = pd.DataFrame()
	subject_y_data = []
	pickle_folder = f"Files/Out/Pickles/{folder_type}/"
	pickle_filepath = f"{pickle_folder}{pickle_file}.pickle"
	
	# Delete previous files
	if os.path.isfile(pickle_filepath):
		# not using delete_files fnc because there is another file in this folder that we need to keep
		os.remove(pickle_filepath)
	
	# For each subject
	for subject in subjects:
		subject = subject.loc[:, ~subject.columns.isin(['ID'])]  # Removing the ID to avoid error in the specs functions
		subject.reset_index(inplace=True, drop=True)
		for label in range(1, 13):
			# Get the data for this activity
			activity = subject[subject['Label'] == label]
			
			# Removing the Label to avoid error in the specs functions
			activity = activity.loc[:, ~activity.columns.isin(['Label'])]
			# Change the name of the column to allow the user to repair the specifications on the plot
			# Get the specifications for this activity
			x_data, y_data = get_specs(activity, label, time_window_length, non_overlapping_length)
			
			# Regroup the data
			subject_x_data = pd.concat([subject_x_data, x_data], ignore_index=True)
			subject_y_data = subject_y_data + y_data
	
	# Save the Subject’s DataFrame as a .pickle file
	with open(pickle_filepath, "wb") as f:
		pickle.dump([subject_x_data, subject_y_data], f)
	
	# Execute the protocol of Dimensionality reduction
	print("   - Dimensionality Reduction Protocol")
	new_x_data, new_y_data = dim_reduc_protocol(pickle_filepath=pickle_filepath,
	                                            file_name=f"DimReduc{time_window_length}-{non_overlapping_length}",
	                                            folder_type=folder_type)

	return new_x_data, new_y_data
