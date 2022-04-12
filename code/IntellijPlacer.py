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
        self.ref_bin_objects = []

    def to_pol(self, x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho

    def to_cart(self, theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    def read_pics(self, path, pic_num):
        print(os.getcwd())
        self.get_pics(path, True)

        #find objects
        self.ref_bin_objects = [self.binarize(t) for t in self.pics]

        image = imageio.imread(path + "/small_res/" + str(pic_num) + ".jpg")
        self.img = self.binarize(image)
        #cv.imshow('binarisation', self.img)

    def rotate(self, cnt, angle, center, img_size):
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cont_normal = cnt - [cx, cy]

        coordinates = cont_normal[:, 0, :]
        xs, ys = coordinates[:, 0], coordinates[:, 1]
        thetas, rhos = self.to_pol(xs, ys)

        thetas = np.rad2deg(thetas)
        thetas = (thetas + angle) % 360
        thetas = np.deg2rad(thetas)

        xs, ys = self.to_cart(thetas, rhos)

        cont_normal[:, 0, 0] = xs
        cont_normal[:, 0, 1] = ys

        cont_rotated = cont_normal + [img_size[0] / 2, img_size[1] / 2]
        cont_rotated = cont_rotated.astype(np.int32)

        return cont_rotated

    def get_pics(self, path, is_small = False):
        if is_small and os.path.isdir(path + "/small_res"):
            pics = [imageio.imread(path + "/small_res/" + fname) for fname in os.listdir(path+"/low") if fname.endswith(".jpg")]
            return
        pics = [imageio.imread(path + "/" + fname) for fname in os.listdir(path) if fname.endswith(".jpg")]
        pics = pics[1:]
        if is_small:
            pics = [resize(t, (512, 512 * t.shape[1] / t.shape[0])) for t in pics]
            pics = [t[0: t.shape[0] - 0, 0: t.shape[1] - 0] for t in pics]
            os.mkdir(path+"/small_res")
            for i in range(len(pics)):
                imageio.imwrite(path + "/small_res/" + str(i) + ".jpg", pics[i])

    def cv2_image_from_imageio(self, img):
        return (img * 255).astype(np.uint8)

    def binarize(self, image):
        gray = rgb2gray(image)
        new_edge_map = binary_closing(canny(gray, sigma=0.4), footprint=np.ones((3, 3)))
        new_edge_segmentation = binary_fill_holes(new_edge_map)
        return self.cv2_image_from_imageio(new_edge_segmentation)


    def check_place(self, placement: np.ndarray):
        err_cnt = 0
        for el in placement.flat:
            if el > 255:
                err_cnt += 1
                if err_cnt > placement.shape[0] * placement.shape[1] / 5000:
                    return False
        return True

    def check_shape(self, destination: np.ndarray, objects):
        for o in objects:
            if (o.shape[0] >= destination.shape[0] and o.shape[0] >= destination.shape[1]) or (o.shape[1] >= destination.shape[0] and o.shape[1] >= destination.shape[1]):
                return False
        return True

    def fit_iteration(self, destination: np.ndarray, objects, idx):
        print('  '*idx, idx)
        object = objects[idx]
        rows, cols = object.shape
        angle = 0
        step = 3
        percent = 0.05

        #print("new try")

        while angle <= 360:
            print('  '*idx, 'a =', angle)
            for half_trans_y in range(destination.shape[0] // 2):
                for half_trans_x in range(destination.shape[1] // 2):

                    trans_y = 2 * half_trans_y
                    trans_x = 2 * half_trans_x
                    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
                    rotated_object = cv.warpAffine(object, M, (cols, rows))
                    copy = np.ones(shape = (destination.shape[0] + 2 * object.shape[0], destination.shape[1] + 2 * object.shape[1])) * 255
                    copy[object.shape[0]:object.shape[0] + destination.shape[0], object.shape[1]:object.shape[1] + destination.shape[1]] = destination
                    copy[trans_y + object.shape[0]:trans_y + 2 * object.shape[0],
                    trans_x + rotated_object.shape[1]:trans_x + 2 * rotated_object.shape[1]] += rotated_object
                    if self.check_place(copy):
                        if idx == len(objects) - 1:
                            return copy
                        else:
                            rez = self.fit_iteration(copy, objects, idx + 1)
                            if rez is not None:
                                return rez
                    half_trans_x += int( percent * destination.shape[1]//2)
                half_trans_y += int( percent * destination.shape[0]//2)
            angle += step
        return None


    def try_fit(self):
        contours, hierarchy = cv.findContours(self.img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        counters_modified = []
        res_counters = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)
        rects = []
        max_area_idx = 0
        max_area = cv.contourArea(contours[0])
        for i in range(len(contours)):
            area = cv.contourArea(contours[i])
            if hierarchy[0][i][3] != -1 or area <= 200:
                continue
            if area > max_area:
                max_area, max_area_idx = area, len(rects)

            counters_modified.append(contours[i])
            rects.append(cv.minAreaRect(contours[i]))
            color = (rng.randint(0, 256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(res_counters, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
            box = cv.boxPoints(rects[-1])
            box = np.int0(box)
            cv.drawContours(res_counters, [box], 0, (0, 0, 255), 2)

        #cv.imshow('Contours', res_counters)

        masks = []
        for rec in rects:
            masks.append(np.zeros((int(rec[1][1]), int(rec[1][0]))).astype(np.uint8))
            rotated_cnt = counters_modified[len(masks) - 1]
            rotated_cnt = self.rotate(rotated_cnt, -rec[2], rec[0], rec[1])
            cv.fillPoly(masks[-1], pts=[rotated_cnt], color=(255, 255, 255))

        polygon = masks.pop(max_area_idx)
        polygon = (255 - polygon)
        #cv.imshow("poly", polygon)

        placement = polygon

        if not self.check_shape(placement, masks):
            placement = None
        else:
            i = 0
            placement = self.fit_iteration(placement, masks, 0)

        if placement is None:
            print("No")
        else:
            print("Yes")
            cv.imshow("placement", placement)
        cv.waitKey()
