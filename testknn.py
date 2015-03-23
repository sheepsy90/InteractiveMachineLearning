import random
import unittest
import matplotlib.pyplot as plt

from knn import KNN


class TestKNN(unittest.TestCase):

    def test_basic_behaviour(self):
        pnts_label_1 = [(random.random()*4, random.random()) for i in range(10)]
        pnts_label_2 = [(random.random()+3, random.random()+3) for i in range(10)]

        knn = KNN(2)

        [knn.add_data_point(1, p) for p in pnts_label_1]
        [knn.add_data_point(2, p) for p in pnts_label_2]

        for i in range(50):
            for j in range(50):
                x, y = i / 10.0, j / 10.0

                result = knn.estimate(3, (x,y))

                if result == 1:
                    plt.plot(x, y, 'rx')
                elif result == 2:
                    plt.plot(x, y, 'bx')

        plt.plot([e[0] for e in pnts_label_1], [e[1] for e in pnts_label_1], 'ro')
        plt.plot([e[0] for e in pnts_label_2], [e[1] for e in pnts_label_2], 'bo')


        plt.show()