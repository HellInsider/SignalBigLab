import os
import numpy as np
import imageio
from skimage.transform import resize
from skimage.color import rgb2gray
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_closing
from skimage.feature import canny
import cv2 as cv
import random as rng


class IntellijPlacer:

    def __init__(self):
        self.img = None
        self.pics = []
        self.smallres_dir = '/small_res/'

    def to_polar_coordinated(self, x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho

    def to_cartesian_coordinates(self, theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    def read_pics(self, path, pic_num):
        #print("Reading pics...")
        self.get_pics(path, True)
        image = imageio.imread(path + self.smallres_dir + str(pic_num) + ".jpg")
        self.img = self.binarize(image)


    def rotate(self, cnt, angle, img_size):
        M = cv.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cont_normal = cnt - [cx, cy]

        coordinates = cont_normal[:, 0, :]
        xs, ys = coordinates[:, 0], coordinates[:, 1]
        thetas, rhos = self.to_polar_coordinated(xs, ys)

        thetas = np.rad2deg(thetas)
        thetas = (thetas + angle) % 360
        thetas = np.deg2rad(thetas)

        xs, ys = self.to_cartesian_coordinates(thetas, rhos)

        cont_normal[:, 0, 0] = xs
        cont_normal[:, 0, 1] = ys

        cont_rotated = cont_normal + [img_size[0] / 2, img_size[1] / 2]
        cont_rotated = cont_rotated.astype(np.int32)

        return cont_rotated

    def get_pics(self, path, is_small=False):                # загрузка изображений
        if is_small and os.path.isdir(path + self.smallres_dir):
            return

        pics = [imageio.imread(path + "/" + fname) for fname in os.listdir(path) if fname.endswith(".jpg")]

        if is_small:
            pics = [resize(t, (512, 512 * t.shape[1] / t.shape[0])) for t in pics]              # сжатие изображений
            pics = [t[0: t.shape[0], 0: t.shape[1]] for t in pics]
            os.mkdir(path + self.smallres_dir)
            for i in range(len(pics)):
                imageio.imwrite(path + self.smallres_dir + str(i) + ".jpg", pics[i])

    def cv2_image_from_imageio(self, img):
        return (img * 255).astype(np.uint8)

    def binarize(self, image):
        gray = rgb2gray(image)
        new_edge_map = binary_closing(canny(gray, sigma=0.4), footprint=np.ones((3, 3)))    # бинаризация изображений
        new_edge_segmentation = binary_fill_holes(new_edge_map)
        return self.cv2_image_from_imageio(new_edge_segmentation)

    def check_place(self, main_poly: np.ndarray):           # проверка расположений
        max_err_rate = 0.001

        err_cnt = np.count_nonzero(main_poly > 255)
        if err_cnt > main_poly.shape[0] * main_poly.shape[1] * max_err_rate:
            return False

        return True

    def check_shape(self, destination: np.ndarray, objects):  # Проверка размеров объектов
        for o in objects:
            if (o.shape[0] >= destination.shape[0] and o.shape[0] >= destination.shape[1]) or (
                    o.shape[1] >= destination.shape[0] and o.shape[1] >= destination.shape[1]):
                return False
        return True

    def fit_iteration(self, destination: np.ndarray, objects, idx):     # итерация расположений одного объекта
        #print('  ' * idx, "Object num: ", idx)
        object = objects[idx]
        rows, cols = object.shape
        angle = 0
        step = 3
        percent = 0.05

        while angle in range(0, 360, step):                                                              # поворот
            #print('  ' * idx, 'angle =', angle)
            for half_trans_y in range(destination.shape[0] // 2):
                for half_trans_x in range(destination.shape[1] // 2):                       # перемещение по Х и У

                    trans_y = 2 * half_trans_y
                    trans_x = 2 * half_trans_x

                    Matrix = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
                    rotated_object = cv.warpAffine(object, Matrix, (cols, rows))

                    copy = np.ones(shape=(
                        destination.shape[0] + 2 * object.shape[0], destination.shape[1] + 2 * object.shape[1])) * 255

                    copy[object.shape[0]:object.shape[0] + destination.shape[0],
                        object.shape[1]:object.shape[1] + destination.shape[1]] = destination

                    copy[trans_y + object.shape[0]:trans_y + 2 * object.shape[0],
                        trans_x + rotated_object.shape[1]:trans_x + 2 * rotated_object.shape[1]] += rotated_object

                    if self.check_place(copy):
                        if idx == len(objects) - 1:
                            return copy
                        else:
                            rez = self.fit_iteration(copy, objects, idx + 1)
                            if rez is not None:
                                return rez

                    half_trans_x += int(percent * destination.shape[1] // 2)
                half_trans_y += int(percent * destination.shape[0] // 2)
        return None

    def check_image(self):
        min_area_size = 50
        res = False                         # проверка на возможность располажения всех объектов
        contours, hierarchy = cv.findContours(self.img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_modified = []
        rects = []
        max_area_idx = 0
        max_area = cv.contourArea(contours[0])
        for i in range(len(contours)):
            area = cv.contourArea(contours[i])
            if hierarchy[0][i][3] != -1 or area <= min_area_size:
                continue
            if area > max_area:
                max_area, max_area_idx = area, len(rects)

            contours_modified.append(contours[i])
            rects.append(cv.minAreaRect(contours[i]))

        masks = []
        for rect in rects:
            masks.append(np.zeros((int(rect[1][1]), int(rect[1][0]))).astype(np.uint8))
            rotated_cnt = contours_modified[len(masks) - 1]
            rotated_cnt = self.rotate(rotated_cnt, -rect[2], rect[1])
            cv.fillPoly(masks[-1], pts=[rotated_cnt], color=(255, 255, 255))

        polygon = masks.pop(max_area_idx)
        polygon = (255 - polygon)
        main_poly = polygon

        if not self.check_shape(main_poly, masks):
            main_poly = None
        else:
            i = 0
            main_poly = self.fit_iteration(main_poly, masks, 0)

        if main_poly is None:
            res = False
        else:
            cv.imshow("main_poly", main_poly)
            res = True
        cv.waitKey()
        return res
