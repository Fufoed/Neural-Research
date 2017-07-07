import numpy as np
from numpy import array, dot, exp, random, sum

class functions():
    def sigmoid(activation):
        return 1.0 / (1.0 + exp(-activation))

    def SigmoidDerivative(output):
        return output * (1.0 - output)

    def activate(weights, inputs):
        activation = 0
        for i in range(len(weights)):
            activation += weights[i] * inputs[i]
        return activation

    def Predict(network, row):
        outputs = neuralNetwork.Forwardpropagation(network, row)
        return outputs.index(max(outputs))
