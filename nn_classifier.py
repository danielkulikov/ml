import numpy as np
import sklearn.datasets 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class nn_classifier():
    """
    Represents a three-layer feed-forward neural network meant 
    for classification. 
    """
    def __init__(self, X, t, params):
        self.X = X
        self.t = t
        self.params = params
        self.model = {}
    
    def backprop(self, w1, b1, w2, b2):
        """
        Runs one iteration of backprop. Does the forward pass to compute the 
        activations, then computes the gradients and does the GD updates
        (with weight decay).
        """
        # forward pass - first activation
        z1 = self.X.dot(w1.T) + b1.T
        a1 = self.first_activation(z1)
        
        # second activation
        z2 = a1.dot(w2.T) + b2.T
        a2 = self.second_activation(z2)
        
        # compute cost
        cost = self.compute_loss(a2, self.t)
        
        # backward pass - compute the gradients for the parameters of the model
        i_e = self.initial_error(a2)
        dW2, db2 = self.compute_output_gradients(i_e, a1)
        h_e = self.hidden_error(i_e, w2, b2, a1)
        dW1, db1 = self.compute_hidden_gradients(h_e)
        
        # get epsilon, lambda parameters
        eps, lamb = self.params["eps"], self.params["lambda"]
        
        # apply l2 regularization (weight decay)
        dW2 += lamb * w2.T
        dW1 += lamb * w1.T
 
        # apply gradient descent parameter updates     
        w1 += -eps * dW1.T
        b1 += -eps * db1.T
        w2 += -eps * dW2.T
        b2 += -eps * db2.T

        return cost, w1, b1, w2, b2

    def train(self):
        """
        Trains the network by running the backpropagation algorithm for the 
        set number of epochs in params. 
        """
        hidden_size, output_size, num_epochs = self.params["h_size"], \
            self.params["o_size"], self.params["num_epochs"]
        
        # initialize weights to small random numbers, biases to 0
        w1 = np.random.randn(hidden_size, self.X.shape[1])
        b1 = np.zeros((hidden_size, 1))
        w2 = np.random.randn(output_size, hidden_size)
        b2 = np.zeros((output_size, 1))
        
        for i in range(0, num_epochs):
            # do a backprop update
            cost, w1, b1, w2, b2 = self.backprop(w1, b1, w2, b2)
            
            # epoch check and print current cost
            if (i % 1 == 0):
                print("Epoch ", i, "cost: ", cost)
                
        self.model = { 'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}
        
    def predict(self, x):
        """
        Do a forward pass to get the softmax values of some test-set, 
        then argmax to predict. 
        """
        w1, b1, w2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # one forward pass to get the softmax
        z1 = x.dot(w1.T) + b1.T
        a1 = self.first_activation(z1)
        z2 = a1.dot(w2.T) + b2.T
        probs = self.second_activation(z2)
        # argmax to classify
        return np.argmax(probs, axis=1)
    
    def first_activation(self, z1):
        """ 
        Computes the selected activation function value for the hidden 
        layer in params.
        """
        first_act = self.params["act"][0]
        if first_act == "tanh":
            return np.tanh(z1)
        if first_act == "softmax":
            return stable_softmax(z1)
        
    def second_activation(self, z2):
        """ 
        Computes the selected activation function value for the output 
        layer in params.
        """
        second_act = self.params["act"][1]
        if second_act == "tanh":
            return np.tanh(z2)
        if second_act == "softmax":
            return stable_softmax(z2)
        
    def initial_error(self, a2):
        """
        Computes the gradient dL/dz2 
        """
        second_act = self.params["act"][1]
        loss = self.params["loss"]
        if(second_act == "softmax" and loss == "ce"):
            return a2 - self.t
        
    def hidden_error(self, i_e, w2, b2, a1):
        """
        Computes the gradient dL/dz1 
        """
        first_act = self.params["act"][0]
        if first_act == "tanh":
            h_e = i_e.dot(w2) * (1 - np.power(a1, 2))
        return h_e
        
    def compute_output_gradients(self, i_e, a1):
        """
        Computes the gradients for the output layer parameters, using the 
        chain rule.
        """
        dW2 = (a1.T).dot(i_e)
        db2 = np.sum(i_e, axis=0, keepdims=True)
        return dW2, db2
    
    def compute_hidden_gradients(self, h_e):
        """
        Computes the gradients for the hidden layer parameters, using the 
        chain rule.
        """
        first_act = self.params["act"][0]
        if first_act == "tanh":
            dW1 = np.dot(self.X.T, h_e)
            db1 = np.sum(h_e, axis=0, keepdims=True)
        return dW1, db1
            
    def compute_loss(self, p, t):
        """
        Computes the loss (using the selected loss function in params).
        """
        loss = self.params["loss"]
        if loss == "ce":
            return cross_entropy(p, t)

def cross_entropy(predictions, targets):
    """
    Computes the cross-entropy loss of a given predictions/targets tuple.
    """
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce

def stable_softmax(x):
    """
    An overflow/underflow stable implementation of softmax (unused but welcome)
    """
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax

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
    
    # one-hot encode the labels for the training and test sets
    y_test_new = np.zeros((y_test.size, y_test.max()+1))
    y_test_new[np.arange(y_test.size), y_test] = 1
    y_test = y_test_new
    y_train_new = np.zeros((y_train.size, y_train.max()+1))
    y_train_new[np.arange(y_train.size), y_train] = 1
    y_train = y_train_new
    
    # set up the parameters for the model
    params = {}
    params["i_size"], params["h_size"], params["o_size"] = X.shape[0], 35, 10
    params["act"], params["loss"] = ["tanh", "softmax"], "ce"
    params["eps"], params["num_epochs"], params["lambda"] = 0.001, 2000, 0.1
    
    # set up a ffnn with sigmoid activation, cross-entropy loss to recognize MNIST digits
    mnist_classifier = nn_classifier(X_train, y_train, params)

    # train the network
    mnist_classifier.train() 

    # make predictions on the test-set
    # then, generate a confusion matrix using seaborn's heatmap()
    results = mnist_classifier.predict(X_test)
    
    cm = confusion_matrix(y_test_temp, results)
    ax = plt.axes()
    sb.heatmap(cm, ax = ax)
    ax.set_title('MNIST Classification: Three-Layer FFNN with Tanh + Softmax + Cross-enropy')
    plt.show()
    
  