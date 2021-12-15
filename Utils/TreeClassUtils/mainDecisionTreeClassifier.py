import pickle

import numpy as np

import utilDecisionTreeClassification
import utilPerformanceComputation

########################################################################################################################
#                                    User Settings for the Decision Tree Classifier                                    #
########################################################################################################################

# Define the path of the dataset
dataset_path_name = "..."

# Create a class object that define parameters of the decision tree classifier
decision_tree_parameters = utilDecisionTreeClassification.DecisionTreeParameters()

""" Function to measure the quality of a split
    Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. 
    Note: this parameter is tree-specific. """
decision_tree_parameters.criterion = 'entropy'

""" The strategy used to choose the split at each node. 
    Supported strategies are “best” to choose the best split and “random” to choose the best random split."""
decision_tree_parameters.splitter = 'best'

""" The maximum depth of the tree.
    The choices are :
                    If None, then nodes are expanded until all leaves are pure or until all leaves contain less than 
                        min_samples_split samples. """
decision_tree_parameters.max_depth = None

""" The minimum number of samples required to split an internal node :
    The choices are :
                      If int, then consider min_samples_leaf as the minimum number.
                      If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are 
                        the minimum number of samples for each node."""
decision_tree_parameters.min_samples_split = 2

""" The minimum number of samples required to be at a leaf node.
    The choices are :
                      If int, then consider min_samples_split as the minimum number.
                      If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the 
                        minimum number of samples for each split. """
decision_tree_parameters.min_samples_leaf = 1

""" The number of features to consider when looking for the best split
    The choices are :
                      If int, then consider max_features features at each split.
                      If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
                      If “auto”, then max_features=sqrt(n_features).
                      If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
                      If “log2”, then max_features=log2(n_features).
                      If None, then max_features=n_features. 

    Note: the search for a split does not stop until at least one valid partition of the node samples is found, even 
    if it requires to effectively inspect more than max_features features."""
decision_tree_parameters.max_feature = 'sqrt'

""" Grow trees with max_leaf_nodes in best-first fashion. 
    If None then unlimited number of leaf nodes. """
decision_tree_parameters.max_leaf_nodes = None

########################################################################################################################
#                                  Display Information, Convert and Create Variables                                   #
########################################################################################################################

# Get the training and testing datasets (they should be split before to run this script)
with open(dataset_path_name, 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)  # The order can change according the order that you pick when
                                                       # you saved these variable

print("     The training dataset has : " + str(len(X_train)) + " instances\n")  # It work if it is a list
print("     The testing dataset has : " + str(len(X_test)) + " instances\n")  # Same
print("     All the instances are splitting into " + str(len(np.unique(y_test))) + " classes, which are : \n")  # Same

# Get the classes (unique labels) of the problem from the list "y_train"
class_names = np.unique(y_train)

# Display the list of classes of the problem
for i in range(0, len(class_names), 1):
    print("         - " + class_names[i])

# Convert the list of class names into an array to display results
class_names = np.array(class_names)  # It was useful in a previous version of this code

# Create a class object that define the performances container of the decision tree classifier
performances = utilPerformanceComputation.Performances()

########################################################################################################################
#                                  Execute the Decision Tree Classifier on the Dataset                                 #
########################################################################################################################

print("The decision tree algorithm is executing. Please wait ...")

# Create and train the model of the decision tree
decision_tree_classifier, training_running_time = \
    utilDecisionTreeClassification.train_decision_tree_classifier(X_train, y_train, decision_tree_parameters)

print("The training process of the model of the decision tree took : %.8f second" % training_running_time)

# Test the trained model of the decision tree
y_test_predicted, testing_running_time = \
    utilDecisionTreeClassification.test_decision_tree_classifier(X_test, decision_tree_classifier)

print("The testing process of decision tree took : %.8f second" % testing_running_time)

# Compute the performances of the decision tree classifier
cm = utilPerformanceComputation.compute_performances_for_multiclass(y_test, y_test_predicted, class_names, performances)

# Display the results
utilPerformanceComputation.display_confusion_matrix(performances, class_names)
utilPerformanceComputation.display_features_and_classification_for_dt_classifier(X_test, y_test, class_names,
                                                                                 decision_tree_classifier)
