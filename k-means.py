# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:38:46 2020

@author: Daniel Kulikov

Implementing k-means from scratch with numpy. 
"""
import numpy as np
import sklearn.datasets

class k_means():
    def __init__(self, k, X, num_iter, init_type):
        self.k = k
        self.X = X
        self.num_iter = num_iter
        self.init_type = init_type
        self.centroids = None
        self.labels = np.zeros(self.X.shape[0])
        
    def cluster(self):
        """
        Runs the k-means algorithm until it convergences on the dataset X.
        """
        # initialize the centroids
        self.init_centroids()
        # run the k-means algorithm num_iter times
        itr = 0
        centroids = np.zeros((self.k, self.X.shape[1]))
        dists = np.zeros((self.X.shape[0], self.k))
        while(itr < self.num_iter):
            # do one iteration of k-means
            # get distances of all points to the centroids
            for i in range(self.k):
                dists[:, i] = np.linalg.norm(self.X - self.centroids[i], axis=1).T
            # change class of each datapoint to closest centroid
            mins = np.argmin(dists, axis=1)
            self.labels = mins
            # change each centroid location to the center of mass 
            for j in range(self.k):
                x_k = self.X[np.where(self.labels==j)]
                mean = np.mean(x_k, axis=0)
                centroids[j, :] = mean

            itr += 1
            if(self.num_iter % 1 == 0):
                print(1)
            
    def converged(self, centroids):
        return (centroids == self.centroids).all()
        
    def init_centroids(self):
        """
        Initializes the centroids.
        """
        if (self.init_type == "random"):
            self.init_random()
        if (self.init_type == "++"):
            self.init_plus()

    def init_random(self):
        """
        Initializes the centroids randomly within R^n
        """
        self.centroids = np.zeros((self.k, self.X.shape[1]))
        self.labels = np.zeros(self.X.shape[0])
        mins = np.min(self.X, axis = 1)
        maxes = np.max(self.X, axis = 1)
        for i in range(self.k):
            self.centroids[i, :] = np.random.uniform(low=mins[i], high=maxes[i], size=(1,self.X.shape[1]))
        
    def init_plus(self):
        """
        Initializes the centroids as per the k-means++ algorithm
        """
        
if __name__=="__main__":
    # import MNIST dataset from scikit_learn
    mnist = sklearn.datasets.load_digits()
    X, y = mnist["data"], mnist["target"]
    
    # normalize the image data
    X = X / 255
    
    # shuffle the data-set randomly
    shuffle_index = np.random.permutation(y.shape[0])
    X_new = X[shuffle_index, :] 
    y_new = y[shuffle_index]
    
    km = k_means(10, X, 100, "random")
    km.cluster()

    