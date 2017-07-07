import numpy as np
from numpy import array, dot, exp, random, sum
import Neuron

class layer():
    def __init__(self, index):
        id = index or 0
        self.neurons = []
    def Populate(self, MaxNeurons, MaxInputs):
        for i in range(MaxNeurons):
            _neuron_ = neuron()
            _neuron_.Populate(MaxInputs)
            self.neurons.append(_neuron_)