import time
from typing import Tuple

import pandas as pd

from Utils.Graphics import plot_stats  # Plotting the stats of features
from Utils.Protocol import protocol  # The protocol of the project

from Utils.Utils import data_cleaning  # To clean the whole dataset


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
	print("Starting main")
	start_time = time.time()
	
	####################################################################################################################
	#                                                  USER PARAMETERS                                                 #
	####################################################################################################################
	
	# Path to the training dataset from the Train/Train.py file
	path = "../Files/Docs/MHEALTHDATASET/TrainSet/"
	
	# List of files used as training set
	subjects_files = [f"mHealth_subject{str(i)}" for i in range(1, 10)]
	
	# Variables used in the protocol
	pickle_file = "TrainSet"
	folder_type = "Train"
	
	####################################################################################################################
	#                                 DATA CLEANING                                                                    #
	####################################################################################################################
	print("- Data Cleaning")
	subjects_list_df = data_cleaning(path, subjects_files)
	
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
		plot_stats(subjects_list_df, features=features, save_path="../Files/Out/Stats/Train/")
	
	print("- Launch Protocol")
	x_data, y_data = protocol(subjects_list_df, time_window_length=time_window_length,
	                          non_overlapping_length=non_overlapping_length,
	                          pickle_file=pickle_file, folder_type=folder_type)
	
	end_time = time.time()
	print(f"Total Time = {end_time - start_time}")
	
	return x_data, y_data


if __name__ == "__main__":
	x, y = train(200, 10)
	