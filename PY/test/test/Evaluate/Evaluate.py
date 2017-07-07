import numpy as np
from numpy import array, dot, exp, random, sum
from random import randrange

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
