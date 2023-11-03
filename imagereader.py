import pygame

asurf = pygame.image.load('testimage.png')
bitmap = pygame.surfarray.array2d(asurf)
for arr in bitmap:
    for x in arr:
        print(hex(x))