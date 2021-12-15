import pandas as pd
import numpy as np

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


