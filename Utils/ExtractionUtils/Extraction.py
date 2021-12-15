import numpy as np
import pandas as pd
from scipy.stats import stats


def get_specs(df, time_window_length, non_overlapping_length):
	"""
	Get a dataframe with specifications calculated
	:param pd.DataFrame df: the dataframe used
	:param int time_window_length: The number of date in one window
	:param int non_overlapping_length: The number of different dates between two window following each other
	:return (pd.DataFrame, list): Tuple of the dataframe of extracted specifications and the list of y_data
	"""
	
	label = df['Label'].loc[0]
	df = df.loc[:, df.columns != 'Label']  # Remove the Label column before calculation
	x_data = []
	feature_list = []
	y_data = []
	
	column_names = df.columns
	data_to_use = df.values
	
	for i in range(0, data_to_use.shape[0] - time_window_length, non_overlapping_length):
		x = data_to_use[i:i + time_window_length]
		# Extract features
		vector_features, feature_list = extract_specs(x)
		
		# Add features data to the dataset
		x_data.append(vector_features)
		y_data.append(label)
	
	# Initialize the variable of the new column names
	new_column_names = []
	
	# Create the new column names
	for feature in feature_list:
		
		for column_name in column_names:
			# Create and add the new column name
			new_column_names.append(feature + "_" + column_name)
	
	# Create the DataFrame containing the new dataset
	x_data = pd.DataFrame(x_data, columns=new_column_names)
	
	return x_data, y_data


def get_specs_min(df):
	"""
	Get the min value of the Dataframe
	:param pd.DataFrame df: the currently used DataFrame
	:return: the min value
	:rtype: float
	"""
	return np.min(df, axis=0)


def get_specs_max(df):
	"""
	Get the max value of the Dataframe
	:param pd.DataFrame df: the currently used DataFrame
	:return: the max value
	:rtype: float
	"""
	return np.max(df, axis=0)


def get_specs_mean(df):
	"""
	Get the mean value of the Dataframe
	:param pd.DataFrame df: the currently used DataFrame
	:return: the mean value
	:rtype: float
	"""
	return np.mean(df, axis=0)


def get_specs_std(df):
	"""
	Get the std value of the Dataframe
	:param pd.DataFrame df: the currently used DataFrame
	:return: the std value
	:rtype: float
	"""
	return np.std(df, axis=0)


def get_specs_skewness(df):
	"""
	Get the skewness value of the Dataframe
	:param pd.DataFrame df: the currently used DataFrame
	:return: the skewness value
	:rtype: float
	"""
	return stats.skew(df, axis=0)


def get_specs_kurtosis(df):
	"""
	Get the kurtosis value of the Dataframe
	:param pd.DataFrame df: the currently used DataFrame
	:return: the kurtosis value
	:rtype: float
	"""
	return stats.kurtosis(df, axis=0)


def get_specs_variance(df):
	"""
	Get the variance value of the Dataframe
	:param pd.DataFrame df: the currently used DataFrame
	:return: the variance value
	:rtype: float
	"""
	return np.var(df, axis=0)


def get_specs_ptp(df):
	"""
	Get the peak-to-peak value of the Dataframe
	:param pd.DataFrame df: the currently used DataFrame
	:return: the peak-to-peak value
	:rtype: float
	"""
	return np.ptp(df, axis=0)


def extract_specs(df):
	"""
	Get all specifications and their order
	:param pd.DataFrame df: the currently used DataFrame
	:return: specs, the array of all specs and a list of type of specs
	:rtype: (numpy.array,list)
	"""
	specs = get_specs_min(df)
	specs = np.append(specs, get_specs_max(df))
	specs = np.append(specs, get_specs_mean(df))
	specs = np.append(specs, get_specs_std(df))
	specs = np.append(specs, get_specs_skewness(df))
	specs = np.append(specs, get_specs_kurtosis(df))
	specs = np.append(specs, get_specs_variance(df))
	specs = np.append(specs, get_specs_ptp(df))
	return specs, ["min", "max", "mean", "std", "skew", "kurt", "var", "ptp"]
