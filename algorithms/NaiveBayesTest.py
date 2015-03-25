import random
import unittest
from algorithms.NaiveBayes import NaiveBayes


class TestNaiveBayes(unittest.TestCase):

    def test_with_data_set(self):
        with open("../data_sets/pima-indians-diabetes.data", 'r') as f:
            data = f.read()
            data = data.split("\n")
            data = [d.split(",") for d in data[0:len(data)-1]]
            data = [[float(p) for p in d] for d in data]

        random.shuffle(data)
        train, test = data[:int(len(data)*0.67)], data[int(len(data)*0.67):]

        nb = NaiveBayes()
        for element in train:
            nb.add_data_point(element[-1], element[0:len(element)-1])
        nb.build_model()

        success = 0

        for element in test:
            if int(element[-1]) == nb.predict(element):
                success += 1

        print "Result:", success, len(test), success / float(len(test))
