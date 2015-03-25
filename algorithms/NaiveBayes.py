import random
import unittest
import numpy as np
import math


class NaiveBayes(object):

    def __init__(self):
        self.data_by_class = {}
        self.statistical_model = None

    def add_data_point(self, label, features):
        if label not in self.data_by_class:
            self.data_by_class[label] = []
        self.data_by_class[label].append(features)

    def get_data_points(self):
        return [(k, j) for k, v in self.data_by_class.items() for j in v]

    def clear(self):
        self.data_by_class = {}
        self.statistical_model = None

    def build_model(self):
        # First Step is to calculate mean by features and class
        statistical_attributes = {}

        for key, value in self.data_by_class.items():
            np_matrix = np.array(value)
            means, stds = np_matrix.mean(axis=0, dtype=np.float32), np_matrix.std(axis=0, ddof=1)
            statistical_attributes[int(key)] = [(means[i], stds[i]) for i in range(len(means))]

        self.statistical_model = statistical_attributes

    def calculate_class_probabilities(self, input_vector):
        assert self.statistical_model is not None

        probabilities = {}
        for classValue, classSummaries in self.statistical_model.iteritems():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = input_vector[i]
                try:
                    probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
                except:
                    probabilities[classValue] *= 0
        return probabilities

    def has_model(self):
        return self.statistical_model is not None

    def predict(self, input_vector, probabilistic):
        probs = self.calculate_class_probabilities(input_vector)
        lst = [(k, v) for k, v in probs.items()]
        lst = sorted(lst, key=lambda arg: -arg[1])

        if probabilistic:
            divisior = math.sqrt(sum([e[1]**2 for e in lst]))
            if divisior != 0:
                lst = [[e[0], e[1]/divisior] for e in lst]
                rnd_nr = random.random()
                cum = 0
                for element in lst:
                    cum += element[1]
                    if cum > rnd_nr:
                        return element[0]
                return lst[-1][0]
        return lst[0][0]

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

