# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 2020

@author: Daniel Kulikov
"""
import numpy as np
import sklearn.datasets 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class gda:
    """
    Implementation of GDA (gaussian discriminant analysis) from scratch - for practice.
    -> if we want LDA we construct a single covariance matrix for all classes
    -> if we want QDA we construct a covariance matrix for each class
    -> if we want Gaussian Naive Bayes we have a diagonal covariance matrix (since
    every feature is independent)
    """
    def __init__(self, X, y, cov_type):
        """
        Arguments:
        X: input data
        y: input labels
        params: 
        """
        self.X = X
        self.y = y
        self.num_labels =np.max(self.y)+1
        self.cov_type = cov_type
        
    def get_prior(self):
        """
        Get label_count/total_count as our prior for each class. 
        """
        _, counts = np.unique(self.y, return_counts=True)
        return counts/len(self.y)
    
    def compute_gaussian_pdf_qda(self, x, means, sigmas):
        """
        Computes the conditional probabilities of each data-point for each class
        Returns a numpy array of shape (n, f)
        """
        probs = np.zeros((x.shape[0], self.num_labels))
        constant = np.power((2*np.pi), self.X.shape[1]/2)
        # for each class we compute our conditional probability
        for l in range(self.num_labels):
            const = np.divide(1, np.multiply(constant, np.power(np.linalg.det(sigmas[l, :, :]), 1/2)))
            diff = np.subtract(x, means[l])
            inv = np.linalg.inv(sigmas[l])
            exponent = (-0.5 * ((diff @ inv) * diff)).sum(axis=1).flatten()
            probs[:, l] =  np.exp(exponent)
        return probs
    
    def compute_gaussian_pdf_lda(self, x, means, sigmas):
        pass
        
    def compute_gaussian_pdf_nb(self, x, means, sigmas):
        pass

    def compute_means_and_sigmas_qda(self):
        """
        Computes the mean and distinct covariance matrix of each class for QDA.
        Returns a numpy array of shape (k, f) and a numpy array of shape (k, f, f)
        """
        # set up variable
        mus = np.zeros((self.num_labels, self.X.shape[1]))
        sigmas = np.zeros((self.num_labels, self.X.shape[1], self.X.shape[1]))
        # for each label l
        for l in range(self.num_labels):
            # get all rows with l
            rows = self.X[self.y==l]
            # compute the mean
            mean = np.mean(rows, axis=0)
            sigma = self.cov(rows.T, mean)
            mus[l, :] = mean
            sigmas[l, :, :] = sigma
        self.mus = mus
        self.sigmas = sigmas
        
    def compute_means_and_sigmas_lda(self):
        """
        Computes the means of each class and shared covariance matrix for LDA.
        Returns a numpy array of shape (k, f) and a numpy array of shape (k, f, f)
        """
        pass
        
    def compute_means_and_sigmas_nb(self):
        """
        Computes the means and diagonal covariance matrix for Gaussian Naive Bayes.
        Returns a numpy array of shape (k, f) and a numpy array of shape (k, f, f)
        """
        pass
    
    def cov(self, x, mu):
        """
        Returns a covariance matrix of shape (f,f) for the data points of class l.
        """
        diff = X - mu
        cov = np.dot(diff.T, diff) / X.shape[0]
        return cov + np.eye(X.shape[1]) * 0.001

    def predict_set(self, test_x):
        """
        Compute the conditional probabilities based on our training set, and 
        return the argmax for each test datapoint x_i. 
        """
        # get prior for each class
        priors = self.get_prior()
        # get mean and covariance matrix for each class
        self.compute_means_and_sigmas()
        # use bayes rule to compute the probabilities
        probs = self.compute_gaussian_pdf(test_x, self.mus, self.sigmas)
        # take the largest probability for each test data-point
        preds = np.argmax(probs, axis=1)
        
        return preds
        
    def evaluate_model(self, test_x=None, test_y=None):
        """
        Evaluate the performance of the model on the training set and the test
        set, if given.
        """
        # get performance of training set
        train_preds = self.predict_set(self.X)
        train_count = sum(self.y == train_preds)
        train_acc = train_count/self.X.shape[0]
        print(train_acc)
        
        # get performance of test set
        test_preds = self.predict_set(test_x)
        test_count = sum(test_y == test_preds)
        test_acc = test_count/self.X.shape[0]
        print(test_acc)
        
def mean(x): 
    """
    Gets the mean of x along the columns.
    """
    return np.mean(x, axis = 0)

def covariance_matrix(x):
    """
    Gets the covariance matrix of the features of x.
    """
    return np.cov(x)

def log_gaussian(X, mu, sigma):
    return -(np.sum(np.log(sigma)) + 0.5 * np.sum(((X - mu) / sigma) ** 2)).reshape(-1, 1)

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
    
    # train and test sets of length 1499 and 297 respectively 
    # hard-coded this but can easily just split it up based on percentiles
    X_train = X_new[1:1500, :]
    y_train = y_new[1:1500]
    X_test = X_new[1500:, :]
    y_test = y_new[1500:]
    y_test_temp = y_test
    
    nb = naive_bayes(X_train, y_train, "distinct")
    nb.evaluate_model(X_test, y_test)
    
    
    
    
    
    
    
    