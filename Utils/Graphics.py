import pandas as pd
from matplotlib import pyplot as plt

from Utils.ExtractionUtils.Extraction import get_specs_mean
from Utils.Utils import delete_files


def plot_stats(subjects, features, save_path):
	"""
	Plot the stats of the variables
	:param list subjects: list of all subject dataset
	:param list features: list of all features to plot
	:param str save_path: the path of the folder where the plot are saved
	"""
	# Deleting precedent plot
	delete_files(save_path)
	# List of all possible activity
	activities_name = ["Standing still (1 min) ",
	                   "Sitting and relaxing (1 min) ",
	                   "Lying down (1 min) ",
	                   "Walking (1 min)", "Climbing stairs (1 min) ",
	                   "Waist bends forward (20x) ",
	                   "Frontal elevation of arms (20x)",
	                   "Knees bending (crouching) (20x)",
	                   "Cycling (1 min)",
	                   "Jogging (1 min)",
	                   "Running (1 min)",
	                   "Jump front & back (20x)"]
	
	# Go through each activity
	for i in range(len(activities_name)):
		label = i + 1  # +1 because the 'Label' 0 corresponds to null and was deleted before
		# Then go through each feature in each subject
		tmp_subjects_sample_activity = {}
		
		for feature in features:
			# Get the data of all subject for the current activity
			for subject in subjects:
				# Get the data of this subject for this feature
				tmp_subjects_sample_activity[subject['ID'].loc[0]] = subject[subject['Label'] == label][feature]
				#tmp_subjects_sample_activity[f"mean_{subject['ID'].loc[0]}"] = [get_specs_mean(subject[subject['Label'] == label][feature]) for _ in range(subject.shape[0])]
			
			# Plotting
			to_plot = pd.DataFrame(tmp_subjects_sample_activity)
			to_plot.plot()
			plt.title(label=f"Compare subjects on {activities_name[i]} / {feature}")
			plt.savefig(fname=save_path + f"{label}_{activities_name[i]}_{feature}.png", dpi=100, format='png')
			plt.close()
			