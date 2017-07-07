import numpy as np
from numpy import array, dot, exp, random, sum
from csv import reader

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

