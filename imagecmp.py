import cv2
import numpy as np

image = 'testimage.png'
imagetest = 'testresult.png'
im = cv2.imread(image)
im2 = cv2.imread(imagetest)

im = np.divide(im, 255/2)
im = np.add(im, -1)
best = np.sum(np.multiply(im, im))
print(best)
im2 = np.divide(im2, 255/2)
im2 = np.add(im2, -1)

score = np.sum(np.multiply(im, im2))

print(score)
print(score/best)