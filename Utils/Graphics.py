import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Utils.ExtractionUtils.Extraction import get_specs_mean, get_specs_variance, get_specs_std


def plot_stats(subjects, save_path):
	"""
	Plot the stats of the variables
	:param pd.DataFrame df: The values of the Variable
	:param int label: the number (+1) corresponding to an activity
	:param str column_name: the name of the column used in df
	:param str save_path: the folder path where to save the plot
	"""
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
	
	activities_id = np.unique(df['Label'])
