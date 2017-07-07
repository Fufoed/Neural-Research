import numpy as np
from numpy import array, dot, exp, random, sum

class neuron():
    def __init__(self):
        value = 0
        self.weights = []
        error = 0.0
    def Populate(self, MaxWeights):
        for i in range(MaxWeights):
            self.weights.append(random.random())
