import pandas as pd
import numpy as np
import datetime
from dateutil import parser
from scipy import stats


def get_data(path, sep=',', txt=True) -> pd.DataFrame:
	"""
	Create the dataframe from the txt file and print some information about it
	:param str path: the path to the csv file
	:param char sep: the separation character
	:param bool txt: True to execute all print in this function
	:return: the dataframe of the csv
	:rtype: pd.DataFrame
	"""
	
	# We need to give the names of the columns because there are only data in each file
	names = ["ACC_Chest_X", "ACC_Chest_Y", "ACC_Chest_Z",
	         "ECG_Signal_1", "ECG_Signal_2",
	
	         "ACC_Left_Ankle_X", "ACC_Left_Ankle_Y", "ACC_Left_Ankle_Z",
	         "Gyro_Left_Ankle_X", "Gyro_Left_Ankle_Y", "Gyro_Left_Ankle_Z",
	         "MAG_Left_Ankle_X", "MAG_Left_Ankle_Y", "MAG_Left_Ankle_Z",
	
	         "ACC_Right_Lower_Arm_X", "ACC_Right_Lower_Arm_Y", "ACC_Right_Lower_Arm_Z",
	         "Gyro_Right_Lower_Arm_X", "Gyro_Right_Lower_Arm_Y", "Gyro_Right_Lower_Arm_Z",
	         "MAG_Right_Lower_Arm_X", "MAG_Right_Lower_Arm_Y", "MAG_Right_Lower_Arm_Z",
	
	         "Label"]
	
	# We will work with a part of the data not the whole dataset
	data = pd.read_csv(path, sep=sep, names=names, usecols=[0, 1, 2, 5, 6, 7, 23], engine='python')
	
	# Get the shape of the data
	data_shape = data.shape
	if txt:
		print("\nShapes")
		print(f"There is {str(data_shape[0])} instances and {str(data_shape[1])} features")
		
		# print the features names of the dataset
		print("\nFeatures : ")
		for column_name in data.columns:
			print(f" - {column_name}")
	
	return data


def get_nan(df, on="dataframe") -> np.ndarray:
	"""
	Get all the row which got Nan values
	:param pd.DataFrame df: the data frame used
	:param str on: the type of element the function will be on
	:return: The Dataframe with row that got NaN values
	:rtype: pd.DataFrame
	"""
	res = pd.DataFrame()
	if df.isnull().values.any():
		if on == "dataframe":
			# get the row is a Nan value is found anywhere in the dataframe
			res = df[df.isnull().any(axis=1)]
		else:
			# get the row is a Nan value is found in this column
			res = df[df[on].isna()]
	return res


def get_duplicate(df, on="dataframe", keep=False) -> pd.DataFrame:
	"""
	Get all duplicated rows
	:param pd.DataFrame df: the data frame used
	:param str on: the type of element the function will be on
	:param bool keep: keep parameters for duplicated function
		keep means that all duplicated row will be return
	:return: the dataframe with all duplicated values
	:rtype: pd.DataFrame
	"""
	if on == "dataframe":
		return df[df.duplicated(keep=keep)]
	else:
		# Obtains duplicate row based on one column
		return df[df[on].duplicated(keep=keep)]


def get_specs(df, signal, dates, time_window_length, non_overlapping_length):
	"""
	Get a dataframe with specifications calculated
	:param pd.DataFrame df: the dataframe used
	:param str signal: the column used as a signal
	:param list dates: The two dates between which the values are taken
	:param int time_window_length: The number of date in one window
	:param int non_overlapping_length: The number of different dates between two window following each other
	:return pd.DataFrame: The dataframe of extracted specifications
	"""
	res_data = []
	feature_list = []
	y_data = []
	for i in range(0, df.shape[0] - time_window_length, non_overlapping_length):

		data = df[i:i+time_window_length]
		
		# Get the specifications for this part
		vector_features, feature_list = get_all_specs(data)
		res_data.append(vector_features)
		y_data.append(df['Label'].loc[0])
		
	res_columns = [f"{feature}_{signal}" for feature in feature_list]
	return pd.DataFrame(res_data, columns=res_columns), y_data


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


def get_all_specs(df):
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
