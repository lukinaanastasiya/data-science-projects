import imp
import numpy as np
import pandas as pd
import math
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression, load_digits, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 17

def entropy(y):
    y = y.tolist()
    unique_state = set(y)
    dict_val = {} # Dictionary for unique values and its number
    for i in unique_state:
        dict_val[i] = y.count(i)
    total = len(y)
    ent = 0
    for i in unique_state:
        pi = dict_val[i]/total
        ent += (pi * math.log2(pi))
    return -ent 

def gini(y):
    y = y.tolist()
    unique_state = set(y)
    dict_val = {} # Dictionary for unique values and its number
    for i in unique_state:
        dict_val[i] = y.count(i)
    total = len(y)
    gini = 0
    for i in unique_state:
        pi = dict_val[i]/total
        gini += (pi*pi)
    return 1 - gini

criteria_dict = {'entropy': entropy, 'gini': gini}

def classification_leaf(y):
    return np.bincount(y).argmax() # bincount - count number of enterances of each int value

class Node():
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right
		
class MyDecisionTreeClassifier(BaseEstimator):
    
	def __init__(self, max_depth=np.inf, min_samples_split=2, criterion='gini', debug=False):
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.criterion = criterion
		self.debug = debug
		self._leaf_value = classification_leaf
		self._criterion_function = criteria_dict[self.criterion]
    
	def _functional(self, X, y, feature_idx, threshold):
		mask = X[:, feature_idx] < threshold
		n_obj = X.shape[0]
		n_left = np.sum(mask)
		n_right = n_obj - n_left    
		if n_left > 0 and n_right > 0:
			return self._criterion_function(y) - (n_left / n_obj) * \
                    self._criterion_function(y[mask]) - (n_right / n_obj) * \
                    self._criterion_function(y[~mask])
		else:
			return 0
	
	def _build_tree(self, X, y, depth=1):
		max_functional = 0
		best_feature_idx = None
		best_threshold = None
		n_samples, n_features = X.shape     
        
		if len(np.unique(y)) == 1:
			return Node(labels=y)

        # If the stop criterion is not satisfied, search for the optimal partition
		if depth < self.max_depth and n_samples >= self.min_samples_split:
            # Iterate for all features...
			for feature_idx in range(n_features):
                # and all thresholds for fixed feature
				threshold_values = np.unique(X[:, feature_idx])    
				functional_values = [self._functional(X, y, feature_idx, threshold) for threshold in threshold_values]
                
				best_threshold_idx = np.nanargmax(functional_values)
                    
				if functional_values[best_threshold_idx] > max_functional:
					max_functional = functional_values[best_threshold_idx]
					best_threshold = threshold_values[best_threshold_idx]
					best_feature_idx = feature_idx
					best_mask = X[:, feature_idx] < best_threshold
    
		if best_feature_idx is not None:
            # In case of partition go next recursivelly...
			return Node(feature_idx=best_feature_idx, threshold=best_threshold, 
                        left=self._build_tree(X[best_mask, :], y[best_mask], depth + 1),
                        right=self._build_tree(X[~best_mask, :], y[~best_mask], depth + 1))
		else:
            # else the vertex is a leaf, leave recursion
			return Node(labels=y)
	
	def fit(self, X, y):
		self._n_classes = len(np.unique(y))
		self.root = self._build_tree(X, y)
		return self
        	
	def predict(self, X):
		preds = []
		for x in X:
			node = self.root
			while node.labels is None:
				if x[node.feature_idx] < node.threshold:
					node = node.left
				else:
					node = node.right
			preds.append(self._leaf_value(node.labels))
		return np.array(preds)
        
	def predict_proba(self, X):
		preds = []
		for x in X:
			node = self.root
			while node.labels is None:
				if x[node.feature_idx] < node.threshold:
					node = node.left
				else:
					node = node.right
			for i in range(self._n_classes):
				preds.append([len(node.labels[node.labels == k])/len(node.labels) for k in range(self._n_classes)])
		return preds
				
if __name__ == "__main__":
	X, y = make_classification(n_features=2, n_redundant=0, n_samples=400, random_state=RANDOM_STATE)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
	
	clf = MyDecisionTreeClassifier(max_depth=4, criterion='gini')
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	prob_pred = clf.predict_proba(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy:", accuracy)
	
	clf_2 = DecisionTreeClassifier(max_depth=4, criterion='gini')
	clf_2.fit(X_train, y_train)
	y_pred = clf_2.predict(X_test)
	prob_pred = clf_2.predict_proba(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy:", accuracy)
        