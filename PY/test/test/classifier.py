import numpy as np
from numpy import array, dot, exp, random, sum
from random import seed
from csv import reader
from random import randrange

class neuron():
    def __init__(self):
        value = 0
        self.weights = []
        error = 0.0
    def Populate(self, MaxWeights):
        for i in range(MaxWeights):
            self.weights.append(random.random())

class neuralNetwork():

    def InitializeNetwork(input, hiddens, output):
        network = list()
        index = 0
        prevNeuron = 0
        head = layer(index)
        head.Populate(input, prevNeuron)
        prevNeuron = input
        network.append(head)
        index += 1
        for i in range(len(hiddens)):
            body = layer(index)
            body.Populate(hiddens[i], prevNeuron)
            prevNeuron = hiddens[i]
            network.append(body)
            index += 1
        tail = layer(index)
        tail.Populate(output, prevNeuron)
        network.append(tail)
        return network

    def Forwardpropagation(network, index):
        inputs = index
        for layer in network:
            new_inputs = []
            for neuron in layer.neurons:
                activation = functions.activate(neuron.weights, inputs)
                neuron.value = functions.sigmoid(activation)
                new_inputs.append(neuron.value)
            inputs = new_inputs
        return inputs

    def Backpropagation(network, DesiredOut):
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


    def UpdateWeights(network, index, LearningRate):
        for i in range(len(network)):
            inputs = index[:-1]
            print(inputs);             
            if i != 0:
                inputs = [neuron.value for neuron in network[i - 1].neurons] 
            for neuron in network[i].neurons:
                for j in range(len(inputs)):
                    neuron.weights += LearningRate * neuron.error * inputs[j]
                neuron.weights += LearningRate * neuron.error

    def Train(network, train, LearningRate, Epochs, outputs):
          for epoch in range(Epochs):
              ErrorSum = 0
              for index in train:
                  output = neuralNetwork.Forwardpropagation(network, index)
                  expected = [0 for i in range(outputs)]
                  expected[index[-1]] = 1
                  ErrorSum += sum([(expected[i] - output[i])**2 for i in range(len(expected))])
                  neuralNetwork.Backpropagation(network, expected)
                  neuralNetwork.UpdateWeights(network, index, LearningRate)
              print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, LearningRate, ErrorSum)) 

class layer():
    def __init__(self, index):
        id = index or 0
        self.neurons = []
    def Populate(self, MaxNeurons, MaxInputs):
        for i in range(MaxNeurons):
            _neuron_ = neuron()
            _neuron_.Populate(MaxInputs)
            self.neurons.append(_neuron_)

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

class evaluate():
    def CrossValidation(dataset, folds):
        datasplit = list()
        datacopy = list(dataset)
        size = int(len(dataset) / folds)
        for i in range(folds):
            fold = list()
            while len(fold) < size:
                index = randrange(len(datacopy))
                fold.append(datacopy.pop(index))
            datasplit.append(fold)
        return datasplit

    def Accuracy(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def EvaluateAlgorithm(dataset, algorithm, Maxfolds, *args):
        folds = evaluate.CrossValidation(dataset, Maxfolds)
        scores = list()
        for fold in folds:
            TrainSet = list(folds)
            TrainSet.remove(fold)
            TrainSet = sum(TrainSet, [])
            TestSet = list()
            for row in fold:
                RowCopy = list(row)
                TestSet.append(RowCopy)
                RowCopy[-1] = None
            predicted = algorithm(TrainSet, TestSet, *args)
            actual = [row[-1] for row in fold]
            accuracy = evaluate.Accuracy(actual, predicted)
            scores.append(accuracy)
        return scores

class datasetManager():
    def Load(filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv = reader(file)
            for item in csv:
                if not item:
                    continue
                dataset.append(item)
        return dataset
    
    def ConvertToInt(dataset, column):
        class_values = [row[column] for row in dataset]
        print(class_values)
        unique = set(class_values)
        print(unique)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
             row[column] = lookup[row[column]] 
        return lookup

    def ConvertToFloat(dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())

    def MinMax(dataset):
        minmax = list()
        stats = [[min(column), max(column)] for column in zip(*dataset)]
        return stats
    
    def NormalizeDataset(dataset, minmax):
        for row in dataset:
            for i in range(len(row)-1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
print(n_inputs);
n_outputs = len(set([row[-1] for row in dataset]))
print(n_outputs);
network = neuralNetwork.InitializeNetwork(n_inputs, [2], n_outputs)
neuralNetwork.Train(network, dataset, 0.5, 20, n_outputs)
for i in network:
	for j in i.neurons:print(j.weights)











 





