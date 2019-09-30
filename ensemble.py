from scipy.stats import mode
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

class AdaBoost:
	def __init__(self, n_estimators, max_depth):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
	
	def sampling(self,x,w):
		#mengembalikan index hasil bootstrap sampling
		idx = np.random.choice(x.index, size=len(x), replace=True, p =w)
		return idx
	
	def fit(self, X_train, Y_train):
		self.models=[]
		self.w_models=[]
		self.transform_y = False
		#inisialisasi bobot
		n = len(X_train)
		w = np.ones(n)/n
		
		if 1 in Y_train.unique() and 0 in Y_train.unique():
			Y_train = Y_train.apply(lambda x:-1 if x == 0 else 1)
			self.transform_y = True
		
		for i in range(self.n_estimators):
			while True:
				idx = self.sampling(X_train, w)
				x_sampling = X_train.loc[idx,:]
				y_sampling = Y_train[idx]
			
				dt = DecisionTreeClassifier(max_depth=self.max_depth, criterion='entropy')
				dt.fit(x_sampling, y_sampling)
				dt_prediction = dt.predict(X_train)

				#yang disalah klasifikasikan bernilai 1, yang benar 0
				miss_classified = [int(x) for x in (dt_prediction != Y_train)]
				error_rate = np.dot(miss_classified, w)
				if error_rate >= 0.5:
					continue
				else:
					break
			
			alpha = 0.5 * np.log( (1-error_rate)/error_rate )
			alpha_sign = [x if x==1 else -1 for x in miss_classified]
			w = np.multiply(w, np.exp( [x*alpha for x in alpha_sign] ))
			w /= np.sum(w) #normalisasi 
			
			self.models.append(dt)
			self.w_models.append(alpha)
	
	def predict(self, X_test):
		pred_m = np.zeros(len(X_test))
		for weight, clf in zip(self.w_models, self.models):
			temp = weight * clf.predict(X_test)
			pred_m = pred_m + temp
		
		if self.transform_y:
			temp = np.sign(pred_m)
			return temp.apply(lambda x: 0 if x == -1 else 1)
		else:
			return np.sign(pred_m)
			
class GradientTreeBoosting:
	
	def __init__(self,n_estimators, shrinkage_parameter, max_depth):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.shrinkage_parameter = shrinkage_parameter
		
	def fit(self, X_train, Y_train):
		self.regressors = []
		self.init_log_odd = 0
		self.transform_y = False
		
		if 1 in Y_train.unique() and -1 in Y_train.unique():
			Y_train = Y_train.apply(lambda x:0 if x == -1 else 1)
			self.transform_y = True
		
		f0 = np.log(np.sum(Y_train == 1)/np.sum(Y_train == 0))
		self.init_log_odd = f0
		
		current_log_odds = pd.Series(f0, index=Y_train.index)
		current_gradient = Y_train - expit(current_log_odds.ravel())
		
		#current_gradient = Y_train - (np.exp(current_log_odds)/(1+np.exp(current_log_odds))) 
		for i in range(self.n_estimators):	
			rt = DecisionTreeRegressor(max_depth=self.max_depth)
			rt.fit(X_train, current_gradient)	
			terminal_regions = rt.apply(X_train).copy()
			for leaf in np.where(rt.tree_.children_left == TREE_LEAF)[0]:
				terminal_region = np.where(terminal_regions == leaf)[0]
				residual = current_gradient.take(terminal_region, axis=0)
				y = Y_train.take(terminal_region, axis=0)
				numerator = np.sum(residual)
				denominator = np.sum((y-residual)*(1-y+residual))

				if abs(denominator) < 1e-150:
					rt.tree_.value[leaf, 0, 0] = 0.0
				else:
					rt.tree_.value[leaf, 0, 0] = numerator / denominator
			
			self.regressors.append(rt)
			current_log_odds +=  (self.shrinkage_parameter * rt.tree_.value[:,0,0].take(terminal_regions, axis=0))		
			current_gradient = Y_train - expit(current_log_odds.ravel()) 
	
	def raw_predict(self, X_test):
		""" menghitung log odd setuap X_test """
		log_odds = pd.Series(self.init_log_odd, index=X_test.index) 
		#p = self.regressors[0].predict(X_test)
		for regressor in self.regressors:
			log_odds += self.shrinkage_parameter * regressor.predict(X_test)
			
		return log_odds	
	
	def predict_proba(self, X_test):
		""" menghitung prob setiap X_test """
		raw = self.raw_predict(X_test)
		return expit(raw)
	
	def predict(self, X_test):
		prob = self.predict_proba(X_test)
		#p = prob.apply(lambda x: 1 if x >= 0.5 else 0).copy()
		if self.transform_y:
			p = prob.apply(lambda x: 1 if x >= 0.5 else -1)			
		else:
			p = prob.apply(lambda x: 1 if x >= 0.5 else 0)
		return p
	
	def _check_params(self):
		if self.n_estimators <= 0:
			raise ValueError("n_estimators must be greater than 0 but was %r" % self.n_estimators)
			
		if not (0.0 < self.shrinkage_parameter < 1.0):
			raise ValueError("alpha must be in (0.0, 1.0) but was %r" % self.shrinkage_parameter)	

class RandomForest:
	
	def __init__(self, n_estimators=100, n_features=None, max_depth=1):
		self.n_estimators = n_estimators
		self.n_features = n_features
		self.max_depth = max_depth
		
	def sampling(self, x):
		idx = np.random.choice(x.index, size=len(x), replace=True)
		return idx
	
	def fit(self, X_train, Y_train):
		self.trees = []
		self.transform_y = False
		#self.classes = []
		
		if self.n_features is None:
			self.n_features = "log2"
		
		for _ in range(self.n_estimators):
			dt = DecisionTreeClassifier(max_depth=self.max_depth, max_features = self.n_features, criterion='entropy')
			idx = self.sampling(X_train)
			x_sampling = X_train.loc[idx,:]
			y_sampling = Y_train[idx]
			
			dt.fit(x_sampling, y_sampling)
			self.trees.append(dt)
	
	def check_class(self):
		"""
			return consistent or not and order of the classes in the tree
		"""
		#np.zeros([baris,kolom])
		kelas = np.zeros([0, 2])
		for tree in self.trees:
			kelas = np.append(kelas, [tree.classes_], axis=0 )
		konsisten = np.all(kelas == kelas[0,:])
		order = kelas[0,:]
		return  konsisten, order
		
	def class_order(self):
		k,o = self.check_class()
		if k:
			return o
		else:
			raise Exception('Classes in trees not consistent')
	
	def predict_proba(self, X_test):
		temp = [0,0]
		for pohon in self.trees:
			temp = temp + pohon.predict_proba(X_test)			
		temp /= self.n_estimators
		return temp
		
	def predict(self, X_test):
		proba = self.predict_proba(X_test)
		#self.classes_.take(np.argmax(proba, axis=1), axis=0)
		o = self.class_order()
		p = o.take(np.argmax(proba, axis=1), axis=0)		
		return p
		
	def predict_voting(self, X_test):
		tmp = []
		for m in self.trees:
			p = m.predict(X_test)
			tmp.append(p)
		modes, count = mode(tmp, axis=0)
		o = self.class_order()
		#o.take(modes[0], axis=0)
		return o.take(modes[0], axis=0)
		
