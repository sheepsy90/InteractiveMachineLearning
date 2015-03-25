import random
import unittest

from algorithms.KNearestNeighbour import KNN


class TestKNN(unittest.TestCase):

    def test_basic_behaviour(self):
        with open("../data_sets/pima-indians-diabetes.data", 'r') as f:
            data = f.read()
            data = data.split("\n")
            data = [d.split(",") for d in data[0:len(data)-1]]
            data = [[float(p) for p in d] for d in data]

        random.shuffle(data)
        train, test = data[:int(len(data)*0.67)], data[int(len(data)*0.67):]

        pnts_label_0 = [p[0:len(p)-1] for p in train if int(p[-1]) == 0]
        pnts_label_1 = [p[0:len(p)-1] for p in train if int(p[-1]) == 1]

        knn = KNN(2)

        [knn.add_data_point(1, p) for p in pnts_label_0]
        [knn.add_data_point(2, p) for p in pnts_label_1]


        for k in [1, 3, 5, 7, 9, 11, 13, 15]:
            success = 0
            for data_element in test:
                pnt, label = data_element[0: len(data_element)-1], data_element[-1]
                label += 1

                if int(label) == knn.estimate(k, pnt):
                    success += 1

            print "Result for k={}:".format(k), success, len(test), success / float(len(test))



