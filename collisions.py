from GameObject import GameObject
import pygame


class Collisions:
    def IsColliding(self, object_1: GameObject, object_2: GameObject):
        x1 = object_1.x
        y1 = object_1.y
        w1 = object_1.width
        h1 = object_1.height

        x2 = object_2.x
        y2 = object_2.y
        w2 = object_2.width
        h2 = object_2.height

        rect1 = pygame.Rect(x1, y1, w1, h1)
        rect2 = pygame.Rect(x2, y2, w2, h2)
        if rect1.colliderect(rect2):
            return True
        else:
            return False