import pygame
import math
from algorithms.NaiveBayes import NaiveBayes

#defining color variables
BLACK = (0, 0, 0)
WHITE = (190, 190, 190)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0,255)
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
pygame.display.set_caption("Naive Bayes")
font = pygame.font.Font(None, 36)

#setting fps variable
clock = pygame.time.Clock()

# State Variables
done = False


naive_bayes_storage = NaiveBayes()
draw_area = False
draw_cursor = False

while not done:
    pos = pygame.mouse.get_pos()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            done = True

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_c:
                naive_bayes_storage.clear()

            if event.key == pygame.K_1:
                naive_bayes_storage.add_data_point(0, pos)
                naive_bayes_storage.build_model()
                break
            if event.key == pygame.K_2:
                naive_bayes_storage.add_data_point(1, pos)
                naive_bayes_storage.build_model()
                break
            if event.key == pygame.K_3:
                naive_bayes_storage.add_data_point(2, pos)
                naive_bayes_storage.build_model()
                break
            if event.key == pygame.K_4:
                naive_bayes_storage.add_data_point(3, pos)
                naive_bayes_storage.build_model()
                break

            if event.key == pygame.K_m:
                break

            if event.key == pygame.K_a:
                draw_area = not draw_area
                break
            if event.key == pygame.K_l:
                draw_cursor = not draw_cursor
                break

    screen.fill(BLACK)

    if draw_area:
        for i in range(0, size[0], 20):
            for j in range(0, size[0], 20):
                x,y = i+10, j+10
                result = naive_bayes_storage.predict((i, j))

                if result is not None and not math.isnan(result):
                    pygame.draw.circle(screen, colour_by_class[result], (i, j), 1)


    data_points = naive_bayes_storage.get_data_points()
    # First draw all points on screen
    for pnt in data_points:
        pygame.draw.circle(screen, colour_by_class[pnt[0]], pnt[1], 5)

    text = font.render(str("FPS: " + str(int(clock.get_fps()))), True, WHITE)
    screen.blit(text, (700,10))

    if draw_cursor:
        # Estimate the mouse cursors colour
        result = naive_bayes_storage.predict(pos)

        if result is not None:
            pygame.draw.circle(screen, colour_by_class[result], pos, 8)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()