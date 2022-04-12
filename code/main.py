
import cv2 as cv
import os
import imageio
from IntellijPlacer import IntellijPlacer

if __name__ == '__main__':
    placer = IntellijPlacer()
    #load objects
    print(os.getcwd())
    placer.read_pics('dataset', 1)
    placer.try_fit()




