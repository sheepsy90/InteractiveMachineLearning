import time

__author__ = 'rkessler'
import pygame
import math
from knn import KNN

#defining color variables
BLACK = (0, 0, 0)
WHITE = (190, 190, 190)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GREY = (150, 150, 150)
BLUE = (0, 135,255)
YELLOW = (255, 255, 0)

colour_by_class = {
    0: RED,
    1: GREEN,
    2: BLUE,
    3: YELLOW
}

#window settings
size = (800, 800)
pygame.init()
screen = pygame.display.set_mode(size)
pygame.display.set_caption("K-Means")
font = pygame.font.Font(None, 36)

#setting fps variable
clock = pygame.time.Clock()

# State Variables
done = False
k_mean_started = False

kmeans_state = 0


def mean(lst):
    xs = [e[0] for e in lst]
    ys = [e[1] for e in lst]

    meanxs = sum(xs) / float(len(xs))
    meanys = sum(ys) / float(len(ys))

    return int(meanxs), int(meanys)



class KMeanStorage():

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
        return self.centroids#

    def set_new_centers(self, new_centroids):
        self.centroids = new_centroids

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

kmean_storage = KMeanStorage()

while not done:
    pos = pygame.mouse.get_pos()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            done = True

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_c:
                k_mean_started = False
                kmean_storage.clear()
            if event.key == pygame.K_1:
                kmean_storage.add_data_point(pos)
                break
            if event.key == pygame.K_2:
                kmean_storage.add_centroid(pos)
                break
            if event.key == pygame.K_s:
                k_mean_started = True
                break

    screen.fill(BLACK)

    if k_mean_started:
        if kmeans_state == 0:
            # Assign and draw
            assignments = kmean_storage.kMeans_assignment()
            centroids = kmean_storage.get_centroids()

            for key in assignments:
                lst_pnts = assignments[key]
                centroid_coords = centroids[key]
                for point in lst_pnts:
                    pygame.draw.line(screen, colour_by_class[key], point, centroid_coords, 1)
            kmeans_state = 1

        elif kmeans_state == 1:
            # Move
            assignments = kmean_storage.kMeans_assignment()
            new_centers = {k: mean(v) for k, v in assignments.items()}
            kmean_storage.set_new_centers(new_centers)
            kmeans_state = 0

        time.sleep(2)


    data_pnts = kmean_storage.get_data()
    for pnt in data_pnts:
        pygame.draw.circle(screen, WHITE, pnt, 2)

    cluster_centroids = kmean_storage.get_centroids()
    for pnt in cluster_centroids:
        k, v = pnt, cluster_centroids[pnt]
        pygame.draw.circle(screen, colour_by_class[k], v, 4)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()