import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Utils.Get import get_specs, get_specs_std, get_specs_mean, get_specs_variance

import glob
import os

########################################################################################################################
#                                                   PROTOCOL                                                           #
########################################################################################################################


def protocol(activities, time_window_length, non_overlapping_length, pickle_file, folder):
	# Variables
	df_person = pd.DataFrame()
	y_data = []
	pickle_folder = f"../Files/Out/Pickles/{folder}/"
	pickle_path = f"{pickle_folder}{pickle_file}.pickle"
	time_slider_folder = f"../Files/Out/TimeSlider/{folder}/"
	stats_folder = f"../Files/Out/Stats/{folder}/"

	# Delete previous files
	delete_files(pickle_folder)
	delete_files(time_slider_folder)
	delete_files(stats_folder)
	for activity in activities:
		label = activity['Label'].loc[0]
		for column in activity.columns[:-1]:  # Removing 'Label'
			stats(activity[column], label, column, stats_folder)
			exit()


# 	# Time window
# 	# df_temp,y_temp = get_specs(activity, column, time_window_length, non_overlapping_length)
# 	# df_person = pd.concat([df_person, df_temp], axis=1)
# 	# y_data = y_data + y_temp
# 	...


def stats(df, label, column_name, save_path):
	plt.figure(dpi=100)
	
	#  Get the corresponding activity
	activity = ["Standing still (1 min) ",
	            "Sitting and relaxing (1 min) ",
	            "Lying down (1 min) ",
	            "Walking (1 min)", "Climbing stairs (1 min) ",
	            "Waist bends forward (20x) ",
	            "Frontal elevation of arms (20x)",
	            "Knees bending (crouching) (20x)",
	            "Cycling (1 min)",
	            "Jogging (1 min)",
	            "Running (1 min)",
	            "Jump front & back (20x)"][
		label - 1]  # Minus one because the original list is composed with the first element as an empty element
	to_plot = pd.DataFrame(
		{
			f"{column_name}": df.values.tolist(),
			"mean": [get_specs_mean(df) for _ in range(df.shape[0])]
		}
	)
	print(get_specs_mean(df))
	to_plot.plot()
	plt.title(label=f"Stats for {activity}")
	plt.subplots_adjust(bottom=0.15)
	plt.figtext(0.15, 0.05, f" mean: {get_specs_mean(df)}")
	plt.figtext(0.5, 0.05, f" variance: {get_specs_variance(df)}")
	plt.figtext(0.15, 0.01, f" std: {get_specs_std(df)}")
	plt.savefig(fname=save_path + f"{activity}_{column_name}.png", dpi=100, format='png')
	plt.close()


def delete_files(path):
	"""
	Delete files inside the given folder
	:param str path: the path to a folder to clear
	"""
	if os.path.isdir(path):
		for f in glob.glob(path + "/*"):
			os.remove(f)
			
