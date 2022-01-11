import time
from typing import Tuple

import pandas as pd

from Utils.Graphics import plot_stats  # Plotting the stats of features
from Utils.Protocol import protocol  # The protocol of the project

from Utils.Utils import data_cleaning  # To clean the whole dataset


def test(time_window_length, non_overlapping_length, with_plot_stats=False) -> Tuple[pd.DataFrame, list]:
	"""
	Function for the testing set
	:param int time_window_length: The number of date in one window
	:param int non_overlapping_length: The number of different elements between two window following each other
	:param bool with_plot_stats: True if the user want to create the plots of the stats.
		May cause drastic slowdown performance
	:returns: x_data and y_data, x is the complete dataset and y the list of the class of each x's instance
	:rtype (test_x_data, test_y_data): (pd.DataFrame, list)
	"""
	print("Starting Test")
	start_time = time.time()
	
	####################################################################################################################
	#                                                  USER PARAMETERS                                                 #
	####################################################################################################################
	
	# path to the Testing dataset
	path = "Files/Docs/MHEALTHDATASET/TestSet/"
	# The subject used as Testing subject
	subject = "mHealth_subject10"
	
	# Variables used in the protocol
	pickle_file = "TestSet"
	folder_type = "Test"
	
	####################################################################################################################
	#                                 DATA CLEANING                                                                    #
	####################################################################################################################
	print("- Data Cleaning")
	subjects_list_df = data_cleaning(path, [subject])
	
	# Concat all subjects_list in one dataset
	df_subjects = pd.concat(subjects_list_df, ignore_index=True)
	
	####################################################################################################################
	#                                  PROTOCOLS                                                                       #
	####################################################################################################################
	
	# After the cleaning and the activities' creation, we can launch the protocol
	
	# Get the stats
	if with_plot_stats:
		print("- Start plot Stats")
		features = df_subjects.columns[:-2]  # Get the columns except 'ID' and 'Label'
		plot_stats(subjects_list_df, features=features, save_path="Files/Out/Stats/Test/")
	
	print("- Launch Protocol")
	test_x_data, test_y_data = protocol(subjects_list_df, time_window_length=time_window_length,
	                                    non_overlapping_length=non_overlapping_length,
	                                    pickle_file=pickle_file, folder_type=folder_type)
	
	end_time = time.time()
	print(f"Total Test Time = {end_time - start_time}")
	
	return test_x_data, test_y_data


if __name__ == "__main__":
	x, y = test(200, 10, True)
