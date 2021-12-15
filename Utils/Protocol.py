import pickle

import pandas as pd

from Utils.DimUtils.dimentionalityReduction import dim_reduc_protocol
from Utils.ExtractionUtils.Extraction import get_specs

########################################################################################################################
#                                                   PROTOCOL                                                           #
########################################################################################################################
from Utils.Graphics import plot_stats
from Utils.Utils import delete_files


def protocol(activities, time_window_length, non_overlapping_length, pickle_file, subject, dim_reduc=1):
	"""
	The protocol of the Project
	:param list activities: list of pd.DataFrame
	:param int time_window_length: The number of date in one window
	:param int non_overlapping_length: The number of different elements between two window following each other
	:param str pickle_file: Name of the .pickle file
	:param str subject: Name of the folder (mHealth_subjectX with X between 1-10)
	:param int dim_reduc: the version of the protocol,
	        is it used with for the first time or the second time ( as asked in the pdf)
	"""
	
	# Variables
	x_data = pd.DataFrame()
	y_data = []
	subject_x_data = pd.DataFrame()
	subject_y_data = []
	pickle_folder = f"../Files/Out/Pickles/{subject}/"
	pickle_filepath = f"{pickle_folder}{pickle_file}.pickle"
	time_slider_folder = f"../Files/Out/TimeSlider/DimReduc{dim_reduc}/{subject}/"
	stats_folder = f"../Files/Out/Stats/{subject}/"
	
	# Delete previous files
	delete_files(pickle_folder)
	# delete_files(time_slider_folder)
	delete_files(stats_folder)
	
	# For each activity
	for activity in activities:
		# Get the features of the activity
		x_data, y_data = get_specs(activity, time_window_length, non_overlapping_length)
		
		# Regroup the data
		subject_x_data = pd.concat([subject_x_data, x_data])
		subject_y_data = subject_y_data + y_data
	
	# Save the Subject’s DataFrame as a .pickle file
	with open(pickle_filepath, "wb") as f:
		pickle.dump([subject_x_data, subject_y_data], f)
	#
	# dim_reduc_protocol(pickle_filepath=pickle_filepath,
	#                    file_name=f"DimReduc{time_window_length}-{non_overlapping_length}", subject=subject)
