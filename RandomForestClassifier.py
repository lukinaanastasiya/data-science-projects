import imp
import numpy as np
import math
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification, make_regression, load_digits, load_boston, load_iris

RANDOM_STATE = 17

class MyRandomForestClassifier(BaseEstimator):
	def __init__(self, n_estimators=10, max_depth=10, max_features=10, random_state=RANDOM_STATE):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.max_features = max_features
		self.random_state = random_state
        
		self.trees = []
		self.feat_ids_by_tree = []
        
	def fit(self, X, y):
		for i in range(self.n_estimators):
			np.random.seed(self.random_state + i)
			col_indices = np.random.choice(range(X.columns.shape[0]), size = self.max_features, replace = False)
			self.feat_ids_by_tree.append(col_indices)
			row_indices = np.random.randint(0,len(X), size=len(X))
			sample_X = X.iloc[row_indices, col_indices]
			sample_y = y[row_indices] 
			dt = DecisionTreeClassifier(max_depth=self.max_depth,max_features=self.max_features, random_state=self.random_state)
			dt.fit(sample_X,sample_y)
			self.trees.append(dt)
		return self
    
	def predict_proba(self, X):
		prediction = None
		for ti, tree in enumerate(self.trees):
			indices = self.feat_ids_by_tree[ti]
			sample_X = X.iloc[:,indices]
			if prediction is None:
				prediction = tree.predict_proba(sample_X)
			else:
				prediction += tree.predict_proba(sample_X)
		return prediction/len(self.trees)
	
if __name__ == "__main__":
	# Need to set data

	X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=RANDOM_STATE)

	clf1 = MyRandomForestClassifier()
	clf1.fit(X_train, y_train)
	print(accuracy_score(clf1.predict(X_test), y_test))

	clf2 = RandomForestClassifier()
	clf2.fit(X_train, y_train)
	print(accuracy_score(clf2.predict(X_test), y_test))