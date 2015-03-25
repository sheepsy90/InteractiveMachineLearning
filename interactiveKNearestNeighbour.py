import pygame
import math
from algorithms.KNearestNeighbour import KNN

#defining color variables
BLACK = (0, 0, 0)
WHITE = (190, 190, 190)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0,255)
YELLOW = (255, 255, 0)

colour_by_class = {
    1: RED,
    2: GREEN,
    3: BLUE,
    4: YELLOW
}

#window settings
size = (800, 800)
pygame.init()
screen = pygame.display.set_mode(size)
pygame.display.set_caption("K-Nearest-Neighbour")
font = pygame.font.Font(None, 36)

#setting fps variable
clock = pygame.time.Clock()

# State Variables
done = False
knn_storage = KNN(4)
current_k = 1
draw_area = False
draw_lines = False

while not done:
    pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                done = True
            if event.key == pygame.K_0:
                knn_storage.clear()
            if event.key == pygame.K_1:
                knn_storage.add_data_point(1, pos)
                break
            if event.key == pygame.K_2:
                knn_storage.add_data_point(2, pos)
                break
            if event.key == pygame.K_3:
                knn_storage.add_data_point(3, pos)
                break
            if event.key == pygame.K_4:
                knn_storage.add_data_point(4, pos)
                break
            if event.key == pygame.K_PLUS:
                current_k += 1
                break
            if event.key == pygame.K_MINUS:
                current_k -= 1
                current_k = max(1, current_k)
                break
            if event.key == pygame.K_a:
                draw_area = not draw_area
                break
            if event.key == pygame.K_l:
                draw_lines = not draw_lines
                break

    screen.fill(BLACK)

    if draw_area:
        for i in range(0, size[0], 20):
            for j in range(0, size[0], 20):
                x,y = i+10, j+10
                result = knn_storage.estimate(current_k, (i, j))

                if result is not None and not math.isnan(result):
                    pygame.draw.circle(screen, colour_by_class[result], (i, j), 1)


    data_points = knn_storage.get_data_points()
    # First draw all points on screen
    for pnt in data_points:
        pygame.draw.circle(screen, colour_by_class[pnt[1]], pnt[0], 5)

    # Draw the current k
    text = font.render(str("Curent K: " + str(current_k)), True, WHITE)
    screen.blit(text, (10,10))

    text = font.render(str("FPS: " + str(int(clock.get_fps()))), True, WHITE)
    screen.blit(text, (700,10))

    if draw_lines:
        # Estimate the mouse cursors colour
        result = knn_storage.estimate_and_get_nearest(current_k, pos)

        if result is not None:
            m_label, pnts = result

            if not math.isnan(m_label):
                for element in pnts:
                    dst, label, tpos = element
                    pygame.draw.line(screen, colour_by_class[label], pos, tpos, 2)

                pygame.draw.circle(screen, colour_by_class[m_label], pos, 8)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()