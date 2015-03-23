from collections import Counter
import math


class KNN():

    def __init__(self, num_classes):
        self.num_clases  = num_classes
        self.data_points = []

    def add_data_point(self, label, pnt):
        assert 1 <= int(label) <= self.num_clases
        self.data_points.append((pnt, label))

    def clear(self):
        self.data_points = []

    def calculate_euclidean_distance(self, e1, e2):
        assert len(e1) == len(e2)
        return math.sqrt(sum([(e1[i] - e2[i])**2 for i in range(len(e1))]))

    def estimate(self, k, pnt):
        try:
            distances = [(self.calculate_euclidean_distance(pnt, p[0]), p[1]) for p in self.data_points]
            distances = sorted(distances, key=lambda e: e[0])
            distances = distances[:k]
            distances = [e[1] for e in distances]
            counted = Counter(distances)
            predicted_label = counted.most_common(1)[0][0]
            return predicted_label
        except:
            return None

    def estimate_and_get_nearest(self, k, pnt):
        try:
            distances = [(self.calculate_euclidean_distance(pnt, p[0]), p[1], p[0]) for p in self.data_points]
            distances = sorted(distances, key=lambda e: e[0])
            distances = distances[:k]
            distance_labels = [e[1] for e in distances]
            counted = Counter(distance_labels)
            predicted_label = counted.most_common(1)[0][0]
            return predicted_label, distances
        except:
            return None


    def get_data_points(self):
        return self.data_points


