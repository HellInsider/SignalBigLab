
import cv2 as cv
import os
import imageio
from IntellijPlacer import IntellijPlacer

if __name__ == '__main__':
    placer = IntellijPlacer()
    print(os.getcwd())
    placer.read_pics('..\dataset', 0)   # второй параметр - номер картинки в дирактории. Начинается с 0.
    print(placer.check_image())




