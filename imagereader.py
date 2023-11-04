import pygame

asurf = pygame.image.load('testimage.png')
bitmap = pygame.surfarray.array2d(asurf)
print(bitmap)