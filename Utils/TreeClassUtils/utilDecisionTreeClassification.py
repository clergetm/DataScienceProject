########################################################################################################################
# Author : Julien Maitre                                                                                               #
# Date : 01 - 31 - 2019                                                                                                #
# Version : 0.1                                                                                                        #
########################################################################################################################

""" This file defines the functions for a decision tree classifier. """

import time

import numpy as np

from sklearn.tree import DecisionTreeClassifier

########################################################################################################################
#                            Define the Classes to be Used for the Decision Tree Classifier                            #
########################################################################################################################


class DecisionTreeParameters(object):

    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None):

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


########################################################################################################################
#                         Define the Functions for the Training of the Decision Tree Classifier                        #
########################################################################################################################


def train_decision_tree_classifier(x_train, y_train, decision_tree_parameters):

    print("\nThe decision tree classifier will be created")

    criterion = decision_tree_parameters.criterion
    max_depth = decision_tree_parameters.max_depth
    min_samples_split = decision_tree_parameters.min_samples_split
    min_samples_leaf = decision_tree_parameters.min_samples_leaf
    min_weight_fraction_leaf = decision_tree_parameters.min_weight_fraction_leaf
    max_features = decision_tree_parameters.max_features
    max_leaf_nodes = decision_tree_parameters.max_leaf_nodes

    # Create an instance of the decision tree classifier
    decision_tree_classifier = DecisionTreeClassifier(criterion=criterion,
                                                      max_depth=max_depth,
                                                      min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf,
                                                      min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                      max_features=max_features,
                                                      max_leaf_nodes=max_leaf_nodes)

    print("The decision tree classifier has been created")
    print("The decision tree classifier is training")

    # Get the start time of the training process
    start_time = time.time()

    # Train the model using the training sets
    decision_tree_classifier.fit(x_train, y_train)

    # Get the end time of the training process
    end_time = time.time()

    # Compute the time that the training process took
    running_time = end_time - start_time

    print("The decision tree classifier has done its training process")

    return decision_tree_classifier, running_time


########################################################################################################################
#                      Define the Functions for the Testing of the Decision Tree Classifier                      #
########################################################################################################################


def test_decision_tree_classifier(x_test, decision_tree_classifier):

    print("\nThe decision tree classifier is being tested with the testing set")

    # Get the start time of the testing process
    start_time = time.time()

    # Make predictions using the testing set
    y_test_predicted = decision_tree_classifier.predict(x_test)

    # Get the end time of the testing process
    end_time = time.time()

    # Compute the time that the testing process took
    running_time = end_time - start_time

    print("The decision tree classifier has done its testing process")

    return y_test_predicted, running_time
