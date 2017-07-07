import numpy as np
from numpy import array, dot, exp, random, sum
import Neuron
import Layer
import Functions


class neuralNetwork():
    def __init__(self):
        self.network = list()

    def InitializeNetwork(self, input, hiddens, output):
        index = 0
        prevNeuron = 0
        head = layer(index)
        head.Populate(input, prevNeuron)
        prevNeuron = input
        self.network.append(head)
        index += 1
        for i in range(len(hiddens)):
            body = layer(index)
            body.Populate(hiddens[i], prevNeuron)
            prevNeuron = hiddens[i]
            self.network.append(body)
            index += 1
        tail = layer(index)
        tail.Populate(output, prevNeuron)
        self.network.append(tail)
        return self.network

    def Forwardpropagation(self, network, index):
        inputs = index
        for layer in network:
            new_inputs = []
            for neuron in layer.neurons:
                activation = functions.activate(neuron.weights, inputs)
                neuron.value = functions.sigmoid(activation)
                new_inputs.append(neuron.value)
            inputs = new_inputs
        return inputs

    def Backpropagation(self, network, DesiredOut):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer.neurons)):
                    error = 0.0
                    for neuron in network[i + 1].neurons:
                        error += (neuron.weights[j] * neuron.error)
                    errors.append(error)
            else:
                for j in range(len(layer.neurons)):
                    neuron = layer.neurons[j]
                    errors.append(DesiredOut[j] - neuron.value)
            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                neuron.error = errors[j] * functions.SigmoidDerivative(neuron.value)
                
    def BackSGD(train, test, LearningRate, epochs, hiddens):
        n_inputs = len(train[0]) - 1
        n_outputs = len(set([row[-1] for row in train]))
        network = neuralNetwork.InitializeNetwork(n_inputs, n_hidden, n_outputs)
        neuralNetwork.Forwardpropagation(network, train, l_rate, n_epoch, n_outputs)
        predictions = list()
        for row in test:
            prediction = functions.Predict(network, row)
            predictions.append(prediction)
        return(predictions)


    def UpdateWeights(self, network, index, LearningRate):
        for i in range(len(network)):
            inputs = index[:-1]
            if i != 0:
                inputs = [neuron.value for neuron in network[i - 1].neurons]
            for neuron in network[i].neurons:
                for j in range(len(inputs)):
                    neuron.weights += LearningRate * neuron.error * inputs[j]
                neuron.weights += LearningRate * neuron.error

    def Train(self, network, train, LearningRate, Epochs, outputs):
          for epoch in range(Epochs):
              ErrorSum = 0
              for index in train:
                  output = neuralNetwork.Forwardpropagation(self, network, index)
                  expected = [0 for i in range(outputs)]
                  expected[index[-1]] = 1
                  ErrorSum += sum([(expected[i] - output[i])**2 for i in range(len(expected))])
                  neuralNetwork.Backpropagation(self, network, expected)
                  neuralNetwork.UpdateWeights(self, network, index, LearningRate)
              print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, LearningRate, ErrorSum))  

