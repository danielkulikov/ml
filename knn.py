# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 01:56:37 2020

@author: Daniel
"""

import numpy as np
import sklearn.datasets 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class knn:
    """
    Implementation of a simple k-NN classifier - for practice. No kd-trees or 
    any complexity speed-ups, just simple sorting. 
    """
    def __init__(self, X, y, params):
        self.X = X
        self.y = y
        self.params = params
        self.model = {}
        
    def predict(self, test_x):
        """
        Classify each data-point in the test-set.
        """
        # get k parameter
        k = self.params["k"]
        # get all distances
        dists = self.distance(self.X, test_x)
        # arg-partition to get the k smallest firs
        top = np.argpartition(dists, k, axis=1)[:, :k]
        # get values of top indices - unvectorized, not sure what most
        # efficient way to do this vectorized is - an exercise for later.
        vals = np.zeros(top.shape)
        for i in range(0, dists.shape[0]):
            vals[i, :] = [self.y[top[i, j]] for j in range(0,k)]
        
        # now that we have the correct indices, let's get the actual predictions
        prediction_indices = np.argmax(vals, axis=1)
        preds = np.zeros(prediction_indices.shape)
        # fill in preds array
        for i in range(0, prediction_indices.shape[0]):
            preds[i] = vals[i, prediction_indices[i]]
            
        return preds
        
    def distance(self, D, x):
        if self.params["distance"] == "euclidean":
            return euclidean_distance(D, x)
        elif self.params["distance"] == "cosine":
            return cosine_distance(D, x)
        
def euclidean_distance(D, T):
    """
    Computes the euclidean distance from a test-set T to 
    every point in D.
    """
    return np.array([[ np.linalg.norm(i-j) for j in D] for i in T])
    
def cosine_distance(D, T):
    """
    Computes the cosine distance from a data-point x to 
    every point in D. Implement later if I decide to play around with k-NNs.
    """
    
def setup():
    """
    Sandbox for testing out the FFNN implementation. Loads the data, splits
    it into train/test sets, then trains the model and tests its accuracy. 
    """
    # import MNIST dataset from scikit_learn
    mnist = sklearn.datasets.load_digits()
    X, y = mnist["data"], mnist["target"]
    
    # normalize the image data
    X = X / 255
    
    # shuffle
    shuffle_index = np.random.permutation(y.shape[0])
    X_new = X[shuffle_index, :] 
    y_new = y[shuffle_index]
    
    # train and test sets of length 1499 and 297 respectively 
    # hard-coded this but can easily just split it up based on percentiles
    X_train = X_new[1:1500, :]
    y_train = y_new[1:1500]
    X_test = X_new[1500:, :]
    y_test = y_new[1500:]
    y_test_temp = y_test
    
    params = {}
    params["distance"], params["k"] = "euclidean", 5
    
    # train the simple knn
    k_nn = knn(X_train, y_train, params)
    results = k_nn.predict(X_test)
    
    cm = confusion_matrix(y_test_temp, results)

    ax = plt.axes()
    sb.heatmap(cm, ax = ax)
    ax.set_title('MNIST Classification: k-NN Classifier with k = 3, Normalized Dataset')
    plt.show()
    
    accuracy = np.sum(results == y_test_temp) / results.shape[0]
    print(accuracy)
    
   