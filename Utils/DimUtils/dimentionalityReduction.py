import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

from Utils.Utils import delete_files


def dim_reduc_protocol(pickle_filepath, file_name, subject):
	"""
	Execute the dimensionality reduction protocol to the given pickle file
	:param str pickle_filepath: The pickle file
	:param str file_name: The name given to each plot created and to the pickle file
	:param str subject: the current subject (mHealth_subjectX) with X between 1-10
	"""
	####################################################################################################################
	#                                                 VARIABLES                                                        #
	####################################################################################################################

	pickle_path = f"../Files/Out/Pickles/{subject}/"
	plot_path = f"../Files/Out/PicklesPlots/{subject}/"
	pickle_file = pickle_path + file_name + ".pickle"
	
	# Remove precedent file
	if os.path.isfile(pickle_file):
		os.remove(pickle_file)  # not using delete_files fnc because there is another file in this folder that we need to keep
	delete_files(plot_path)
	
	####################################################################################################################
	#                                                 LOAD THE DATASET                                                 #
	####################################################################################################################
	
	# Load the dataset (you can modify the variables to be load. In this case, we have x an array of the features extracted
	# for each instance and y a list of labels)
	with open(pickle_filepath, 'rb') as file:
		x, y = pickle.load(file)
	
	# Define the number of attribute to select
	attribute_number_to_select = int(len(x.columns) / 2)
	
	####################################################################################################################
	#                                  REDUCE THE DIMENSIONALITY BY SELECTING FEATURES                                 #
	####################################################################################################################
	
	# Define the classifier
	classifier_model = ExtraTreesClassifier(n_estimators=50)
	
	# Train the classifier model to classify correctly the instances into the correct classes
	classifier_model = classifier_model.fit(x, y)
	
	# Get the score of importances for each attribute
	importance_scores = classifier_model.feature_importances_
	
	# Maintenant c'est a votre tour de coder le reste.
	# Le reste doit extraire de x les N meilleurs attributs et afficher un rapport des attributs selectionnes par ordre
	# croissant d'importances. Puis a la fin, vous sauvegarderez le nouveau dataset.
	
	# Sort the features importances and get the ordered indices
	indices = np.argsort(importance_scores)[::-1]
	# Get the best features according to the reduction algorithm
	columns_to_select = x.columns[indices[0:attribute_number_to_select]]
	# Get the new dataset with the selected features
	new_dataset = x[columns_to_select]
	
	#  Plot the results
	plt.bar(x=np.arange(attribute_number_to_select), height=importance_scores[indices[0:attribute_number_to_select]],
	        tick_label=columns_to_select)
	
	plt.title(f"Feature Importances - Sum = {str(np.sum(importance_scores[indices[0:attribute_number_to_select]]))}")
	plt.xlabel(f"Selected features")
	plt.xticks(rotation=90)
	plt.ylabel(f"Importance score")
	plt.tight_layout()
	
	# Save the results
	if os.path.isfile(plot_path + file_name):
		os.remove(plot_path + file_name)
	plt.savefig(plot_path + file_name + ".png")
	
	# Show the results
	plt.show()
	plt.close()
	
	# Save the dataset as Pickle file
	with open(pickle_file, "wb") as f:
		pickle.dump([new_dataset, y], f)
		