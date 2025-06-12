import numpy as np
import pandas as pd
import pygame
import random
import matplotlib.pyplot as plt

pygame.init()

# pygame hello world
def pygame_interface():
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Hello Pygame")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()