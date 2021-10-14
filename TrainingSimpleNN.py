"""
Kleo Purbollari 
A Simple Neuron 

A neuron is the smallest building block of a neural network. 
A neuron takes input, performs mathematical operations on it, and returns an output.

The mathematics operations consist of: 
    1- multiplying each input by a weight
    2- get the sum of the above, add a bias value of b
    3- input the sum above to an activation function
     
"""
import numpy as np

# Use the sigmoid function as the activation function: f(x) = 1 / (1 + e^(-x))
# https://machinelearningmastery.com/a-gentle-introduction-to-sigmoid-function/
# Note: Activation function can be any function that transforms the input value in the range (−∞,+∞) 
# into a value in range (0, 1)
def activfunc(x):
    return 1 / (1 + np.exp(-x))

class SimpleNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # This function does the 3 mathematical operations described in the beginning
    
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias

        return activfunc(total)
"""
# Single Neuron test case
weights = np.array([34, 2]) 
bias = 8.4                   
n = SimpleNeuron(weights, bias)

c = np.array([-2, 33])       
print(n.feedforward(c)) 
"""

# A simple neural network with 2 inputs, a hidden layer with 2 neurons (n1, n2), an output neuron (o)
# For simplicity, each neuron has the same weights and bias
class SimpleNeuralNetwork:
    def __init__(self):
        weights = np.array([2, 11])
        bias = 3
    
        # The Neuron class here is from the part 1
        self.n1 = SimpleNeuron(weights, bias)
        self.n2 = SimpleNeuron(weights, bias)
        self.o = SimpleNeuron(weights, bias)

    def nnfeedforward(self, x):
        n1_out = self.n1.feedforward(x)
        n2_out = self.n2.feedforward(x)
        
        # feed
        result = self.o.feedforward(np.array([n1_out, n2_out]))

        return result
"""
#SimpleNeuralNetwork test case
x = np.array([1, 4])
nn = SimpleNeuralNetwork()
print("Inputting [1,4] into our simpleNeuralNetwork returns: ", nn.feedforward(x))
"""

#derivative of the activation function
def activefunc_d(x):
    f = activfunc(x)
    return f*(1-f)
"""
#Use Mean Squared Error to measure Loss of true (t) values compared to predicted (p) values
def mse(t,p):
    # Note: t and p must be Numpy-defined arrays of the same size
    return ((t-p)**2).mean()
"""
class TrainedNN:
    def __init__(self):
        # Assigned random normalized values of Gausian distribution for biases and weights
        g = np.random.normal
        self.w1 = g()
        self.w2 = g()
        self.w3 = g()
        self.w4 = g()
        self.w5 = g()
        self.w6 = g()

        # Biases
        self.b1 = g()
        self.b2 = g()
        self.b3 = g()
    
    def feedforward(self, x):
       # neuron
        # x is a numpy array with 2 elements.
        n1 = activfunc(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        n2 = activfunc(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o = activfunc(self.w5 * n1 + self.w6 * n2 + self.b3)
        return o
    
    # data is a set of arrays of 2 variables each, to serve is input
    # t stands for true values (0 or 1)
    # the NN will try to minimize loss/error and approximate to 0 or 1 given input
    def train(self, data, v):
        learn_rate = 0.1
        # train in 1000 loops
        for i in range(1000):
            for x, t in zip(data, v):
                # math operations of each neuron + feedforward
                sum_n1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                n1 = activfunc(sum_n1)
                
                sum_n2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                n2 = activfunc(sum_n2)
        
                sum_o = self.w5 * n1 + self.w6 * n2 + self.b3
                o = activfunc(sum_o)
                p = o
                
                # Naming convention: d_L_d_w1 means "partial L / partial w1"
                # partial derivative of Loss/predicted value
                d_L_d_p = -2 * (t - p)
                
                # output neuron o
                d_p_d_w5 = n1 * activefunc_d(sum_o)
                d_p_d_w6 = n2 * activefunc_d(sum_o)
                d_p_d_b3 = activefunc_d(sum_o)
                
                d_p_d_n1 = self.w5 * activefunc_d(sum_o)
                d_p_d_n2 = self.w6 * activefunc_d(sum_o)
                
                # Neuron n1
                d_n1_d_w1 = x[0] * activefunc_d(sum_n1)
                d_n1_d_w2 = x[1] * activefunc_d(sum_n1)
                d_n1_d_b1 = activefunc_d(sum_n1)
                
                # Neuron n2
                d_n2_d_w3 = x[0] * activefunc_d(sum_n2)
                d_n2_d_w4 = x[1] * activefunc_d(sum_n2)
                d_n2_d_b2 = activefunc_d(sum_n2)
                
                # Update weights/bias for each
                # Neuron n1
                self.w1 -= learn_rate * d_L_d_p * d_p_d_n1 * d_n1_d_w1
                self.w2 -= learn_rate * d_L_d_p * d_p_d_n1 * d_n1_d_w2
                self.b1 -= learn_rate * d_L_d_p * d_p_d_n1 * d_n1_d_b1
                
                # Neuron n2
                self.w3 -= learn_rate * d_L_d_p * d_p_d_n2 * d_n2_d_w3
                self.w4 -= learn_rate * d_L_d_p * d_p_d_n2 * d_n2_d_w4
                self.b2 -= learn_rate * d_L_d_p * d_p_d_n2 * d_n2_d_b2
                
                # Neuron o
                self.w5 -= learn_rate * d_L_d_p * d_p_d_w5
                self.w6 -= learn_rate * d_L_d_p * d_p_d_w6
                self.b3 -= learn_rate * d_L_d_p * d_p_d_b3

"""
# Train NN test
# data - each array shows, price in hundred thousand, square footage in thousands
data = np.array([[1, 0.8], [3, 1.6],   [0.8, 0.6],   
  [5, 1.8], [6, 1.7], [0.7, 0.7],])
#t - true values, 0 for apartments, 1 for houses
t = np.array([0, 1, 0, 1, 1, 0,])

nn = TrainedNN()
nn.train(data, t)

test = nn.feedforward(np.array([8,1.75]))
#test = nn.feedforward(np.array([0.9,0.5]))
print(test)
if test>0.6:
    print("Most likely a house")
elif test <0.4:
    print("Most likely an apartment")
else:
    print("Data inconclusive")
"""
