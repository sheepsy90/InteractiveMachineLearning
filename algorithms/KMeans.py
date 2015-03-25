import math


class KMeans():

    def __init__(self):
        self.centroids = {}
        self.data_points = []

    def clear(self):
        self.centroids = {}
        self.data_points = []

    def add_data_point(self, pnt):
        self.data_points.append(pnt)

    def add_centroid(self, pnt):
        if 0 <= len(self.centroids) < 4:
            self.centroids[len(self.centroids)] = pnt

    def get_data(self):
        return self.data_points

    def get_centroids(self):
        return self.centroids

    def set_new_centers(self, new_centroids):
        if str(new_centroids) == str(self.centroids):
            return True
        else:
            self.centroids = new_centroids
            return False

    def kMeans_assignment(self):
        assignments = {}
        for pnt in self.data_points:
            distances = [(self.calculate_euclidean_distance(pnt, self.centroids[k]), k) for k in self.centroids]
            distances = sorted(distances, key=lambda x: x[0])
            assignment = distances[:1][0]

            if assignment[1] not in assignments:
                assignments[assignment[1]] = []

            assignments[assignment[1]].append(pnt)

        return assignments

    def calculate_euclidean_distance(self, e1, e2):
        assert len(e1) == len(e2)
        return math.sqrt(sum([(e1[i] - e2[i])**2 for i in range(len(e1))]))

    def can_be_started(self):
        return len(self.centroids) > 0 and len(self.data_points) > 0