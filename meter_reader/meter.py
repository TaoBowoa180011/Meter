# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os.path as osp
import numpy as np
import math
import cv2
from PaddleOCR.paddleocr_infer import OcrInference
from paddlex.seg import transforms
from sklearn.cluster import KMeans
import paddlex as pdx
import time

# The size of inputting images (METER_SHAPE x METER_SHAPE) of the segmenter,
# also the size of circular meters.
METER_SHAPE = 512
POLAR_IMAGE_WIDTH = 360 * 4
PI = 3.1415926536
ERROR_THRESHOLD = 5
MULTI_METER_CONFIG = []
NUMBERS_BETWEEN_BIG_SCALE = 8
debug = True


def ocr_check(ocr_digit_dict, scale_max):
    checked_ocr_digit_dict = dict()
    for key in ocr_digit_dict.keys():
        value = float(key)
        loc = ocr_digit_dict[key]
        if value <= scale_max:
            checked_ocr_digit_dict[value] = loc

    return checked_ocr_digit_dict


def clustering_digits(dis_array, ocr_digit_dict):
    """

    :param dis_array:
    :param ocr_digit_dict:
    :return: cluster_0 inside
             cluster_1 outside
    """
    cluster_0 = dict()
    cluster_1 = dict()
    keys = list(ocr_digit_dict.keys())
    kmeans = KMeans(n_clusters=2, random_state=0).fit(dis_array)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    if cluster_centers[0] > cluster_centers[1]:
        inside = 1
        outside = 0
    else:
        inside = 0
        outside = 1

    for i, label in enumerate(labels):
        if label == inside:
            cluster_0[keys[i]] = ocr_digit_dict[keys[i]]
        if label == outside:
            cluster_1[keys[i]] = ocr_digit_dict[keys[i]]
    return cluster_0, cluster_1


def distance_from_center_to_ocr(center, ocr_digit_dict):
    digit_num = len(ocr_digit_dict)
    dis_array = np.zeros((digit_num, 1))
    for i, key in enumerate(ocr_digit_dict.keys()):
        location = ocr_digit_dict[key]
        dis = np.linalg.norm(np.array(location) - center).astype(np.uint8)
        dis_array[i] = [dis]
    return dis_array


def Is_big_scale(index, scale_data_list):
    for i in range((-1) * (NUMBERS_BETWEEN_BIG_SCALE // 2), (NUMBERS_BETWEEN_BIG_SCALE // 2) + 1):
        if 0 < index + i < len(scale_data_list):
            if scale_data_list[index + i][1] > scale_data_list[index][1]:
                return False
            return True


def find_meter_center(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 2, 2)
    h, w = gray.shape
    circles = cv2.HoughCircles(gaussian_blur, cv2.HOUGH_GRADIENT, 2, 512,
                               param1=50, param2=30, minRadius=int(min(h, w) / 3), maxRadius=int(max(h, w) / 2))

    circles = np.uint16(np.around(circles))
    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 9)
    #     # draw the center of the circle
    #     cv2.circle(gray, (i[0], i[1]), 2, (0, 0, 255), 8)
    # cv2.imshow('detected circles', gray)
    # cv2.waitKey(0)

    return circles


def ocr_convert(ocr_img, ocr_digit_dict, meter_config, meter_circle):
    ocr_digit_data = list()
    r_y_min = min(meter_circle[0][0][2], min((METER_SHAPE - meter_circle[0][0][1]), meter_circle[0][0][1]))
    r_x_min = min(meter_circle[0][0][2], min((METER_SHAPE - meter_circle[0][0][0]), meter_circle[0][0][0]))
    outer_radius = min(r_y_min, r_x_min)
    if meter_config['type'] == 'single':
        inner_radius = int(outer_radius / 3)
    else:
        inner_radius = int(outer_radius / 4)
    for row in range(POLAR_IMAGE_WIDTH):
        for col in range(inner_radius, outer_radius):
            theta = row
            rho = col
            y_ = rho * math.cos(PI * 2 * theta / POLAR_IMAGE_WIDTH)
            x_ = rho * math.sin(PI * 2 * theta / POLAR_IMAGE_WIDTH)
            y = int(meter_circle[0][0][1] + y_ + 0.5)
            x = int(meter_circle[0][0][0] - x_ + 0.5)

            if ocr_img[y, x] == 3:
                for key in ocr_digit_dict.keys():
                    location = ocr_digit_dict[key]
                    if [x, y] == location:
                        value = float(key)
                        ocr_digit_data.append([row, value])
                        ocr_digit_dict.pop(key)
                        break
    return ocr_digit_data


def polar_convert_1d_data(scale_polar_image):
    """Get two one-dimension data where 0 represents background and >0 represents
       a scale or a pointer from the rectangular meter.

    Args:
        scale_polar_image (np.array): the two-dimension rectangular meter output
            from function polar_create_line_image().

    Return:
            scale_data (np.array): a one-dimension data where 0 represents background and
            >0 represents scales.
            pointer_data (np.array): a one-dimension data where 0 represents background and
            >0 represents a pointer.
    """
    row, col = np.shape(scale_polar_image)
    scale_data = np.zeros(row, dtype=np.uint8)
    pointer_data = np.zeros(row, dtype=np.uint8)
    # Accumulte the number of positions whose label is 1 along the theta axis.
    # Accumulte the number of positions whose label is 2 along the theta axis.
    for i in range(row):
        for j in range(col):
            if scale_polar_image[i, j] == 1:
                pointer_data[i] += 1
            elif scale_polar_image[i, j] == 2:
                scale_data[i] += 1
    return scale_data, pointer_data


def polar_scale_mean_filtration(scale_data):
    """Set the element in the scale data which is lower than its mean value to 0.

    Args:
        scale_data (np.array): the scale data output from function polar_convert_1d_data().
    """
    mean_data = np.mean(scale_data)
    for col in range(POLAR_IMAGE_WIDTH):
        if scale_data[col] < mean_data:
            scale_data[col] = 0


def create_polar_image(meter_image, meter_center, meter_radius):
    """Convert the circular meter from Cylindrical(x,y) into Spherical Coordinates(theta,r)

            Args:
                meter_image (np.array): the label map output from a segmeter for a meter.
                meter_center :
                meter_radius:
            Returns:
                scale_polar_image(np.array):
            """
    r_y_min = min(meter_radius, min((METER_SHAPE - meter_center[1]), meter_center[1]))
    r_x_min = min(meter_radius, min((METER_SHAPE - meter_center[0]), meter_center[0]))
    circle_radius = min(r_y_min, r_x_min)
    scale_polar_image = np.zeros((POLAR_IMAGE_WIDTH, circle_radius), dtype=np.uint8)
    for row in range(POLAR_IMAGE_WIDTH):
        for col in range(circle_radius - 150, circle_radius):
            theta = row
            rho = col
            y_ = rho * math.cos(PI * 2 * theta / POLAR_IMAGE_WIDTH)
            x_ = rho * math.sin(PI * 2 * theta / POLAR_IMAGE_WIDTH)
            y = int(meter_center[1] + y_ + 0.5)
            x = int(meter_center[0] - x_ + 0.5)
            if meter_image[y, x] == 2:
                scale_polar_image[row, col] = meter_image[y, x]
            if meter_image[y, x] == 1:
                scale_polar_image[row, col] = meter_image[y, x]
    if debug:
        cv2.imshow('scale_polar_image', scale_polar_image * 100)
        cv2.waitKey(0)
    return scale_polar_image


def polar_get_meter_reader(scale_data, pointer_data, ocr_result, meter_config):
    scale_flag = False
    pointer_flag = False
    one_scale_start = 0
    # one_scale_end = 0
    one_pointer_start = 0
    # one_pointer_end = 0
    scale_location = list()
    pointer_data_list = list()
    pointer_location = 0
    pointer_data_sum = 0
    result = {}
    for i in range(POLAR_IMAGE_WIDTH - 1):
        if scale_data[i] > 0 and scale_data[i + 1] > 0:
            if not scale_flag:
                one_scale_start = i
                scale_flag = True
        if scale_flag:
            if scale_data[i] == 0 and scale_data[i + 1] == 0:
                one_scale_end = i - 1
                one_scale_location = (one_scale_start + one_scale_end) / 2
                scale_location.append(one_scale_location)
                one_scale_start = 0
                scale_flag = False
        if pointer_data[i] > 0 and pointer_data[i + 1] > 0:
            pointer_data_sum += pointer_data[i]
            if not pointer_flag:
                one_pointer_start = i
                pointer_flag = True
        if pointer_flag:
            if pointer_data[i] == 0 and pointer_data[i + 1] == 0:
                one_pointer_end = i - 1
                pointer_location = (one_pointer_start + one_pointer_end) / 2
                pointer_data_list.append([pointer_location, pointer_data_sum])
                one_pointer_start = 0
                pointer_data_sum = 0
                pointer_flag = False
    if len(pointer_data_list) > 0:
        pointer_data_max = pointer_data_list[0][1]
        pointer_location = pointer_data_list[0][0]
        for p in pointer_data_list:
            if p[1] > pointer_data_max:
                pointer_data_max = p[1]
                pointer_location = p[0]
    scale_data_list = list()

    for s in scale_location:
        scale_data_sum = 0
        for index in range(-2, 2):
            scale_data_sum += scale_data[int(s) + index]
        scale_data_list.append([s, scale_data_sum])
    print('scale_data_list', scale_data_list)

    rectified_ocr = list()
    if len(ocr_result) > 0:
        if meter_config['type'] == 'single':
            for ocr_ in ocr_result:
                if ocr_[1] == 0:
                    rectified_ocr.append([scale_data_list[0][0], ocr_[1]])
                else:
                    scale_max = list()
                    for i in range(len(scale_data_list)):
                        # find the most close big scale to ocr digit
                        if abs(ocr_[0] - scale_data_list[i][0]) < 30:
                            if Is_big_scale(i, scale_data_list):
                                scale_max.append(scale_data_list[i])

                    if len(scale_max) == 1:
                        rectified_ocr.append([scale_max[0][0], ocr_[1]])
                    elif len(scale_max) > 1:
                        max_ = scale_max[0][1]
                        loc_ = scale_max[0][0]
                        for i in range(1, len(scale_max)):
                            if scale_max[i][1] > max_:
                                max_ = scale_max[i][1]
                                loc_ = scale_max[i][0]
                        rectified_ocr.append([loc_, ocr_[1]])
                    else:
                        rectified_ocr.append([ocr_[0], ocr_[1]])
                        print('rectified_ocr', rectified_ocr)

    if debug:
        print('pointer_data_list', pointer_data_list)
        print('pointer_location', pointer_location)

    scale_num = len(scale_location)
    scales = 0
    if pointer_location == 0:
        result = {}
        print("pointer detected failed")
    elif pointer_location < scale_location[0]:
        result = {}
        print('pointer below start point')
    elif pointer_location >= scale_location[0]:
        # pointer in the middle of start and end scales: # at least two scales
        # generae methods for counting scales

        if meter_config['type'] == 'single':
            single_digit_ocr = 0
            # search pointer and ocr digit locations
            if len(rectified_ocr) > 0:
                digit_start_location = scale_location[0]
                digit_end_location = scale_location[-1]
                digit_start_value = meter_config['scale_min']
                digit_end_value = meter_config['scale_max']
                for i in range(len(rectified_ocr)):
                    if rectified_ocr[i][0] < pointer_location and rectified_ocr[i][1] >= digit_start_value:
                        digit_start_location = rectified_ocr[i][0]
                        digit_start_value = rectified_ocr[i][1]
                    if rectified_ocr[i][0] > pointer_location:
                        break
                for j in range(len(rectified_ocr) - 1, -1, -1):
                    if rectified_ocr[j][0] > pointer_location and rectified_ocr[j][1] <= digit_end_value:
                        digit_end_location = rectified_ocr[j][0]
                        digit_end_value = rectified_ocr[j][1]
                    if rectified_ocr[j][0] < pointer_location:
                        break
                if debug:
                    print('(digit_start_location, digit_start_value)', digit_start_location, digit_start_value)
                    print('(digit_end_location, digit_end_value)', digit_end_location, digit_end_value)

                ocr_ratio = (pointer_location - digit_start_location) / (
                        digit_end_location - digit_start_location + 1e-05)
                single_digit_ocr = digit_start_value + (digit_end_value - digit_start_value) * ocr_ratio

                if (digit_start_location in scale_location) and (digit_end_location in scale_location):
                    ocr_start = scale_location.index(digit_start_location)
                    ocr_end = scale_location.index(digit_end_location)
                    if abs((ocr_end - ocr_start) - (digit_end_value - digit_start_value) \
                           / meter_config['scale_value']) <= 2:
                        for s in range(ocr_start, ocr_end):
                            if scale_location[s] <= pointer_location < scale_location[s + 1]:
                                scales_between_ocr = (s - ocr_start) + (pointer_location - scale_location[s]) / (
                                        scale_location[s + 1] - scale_location[s] + 1e-05)
                                single_digit_ocr = digit_start_value + meter_config['scale_value'] * scales_between_ocr

            else:
                # there is no ocr digit found out
                if abs(scale_num - (meter_config['scale_max'] - meter_config['scale_min']) / meter_config[
                    'scale_value']) \
                        <= 3:
                    for i in range(scale_num - 1):
                        if scale_location[i] <= pointer_location < scale_location[i + 1]:
                            scale_ratio = (pointer_location - scale_location[i]) / (
                                    scale_location[i + 1] - scale_location[i] + 1e-05)
                            scales = (i + scale_ratio + 1) * meter_config['scale_value']
                            break
                else:
                    scale_ratio = (pointer_location - scale_location[0]) \
                                  / (scale_location[-1] - scale_location[0] + 1e-05)
                    scales = (meter_config['scale_max'] - meter_config['scale_min']) * scale_ratio
            result = {'scales': scales, 'single_digit_ocr': single_digit_ocr}
        else:
            # meter_config['type']  == 'double'
            if len(ocr_result) == 2:
                cluster_0 = ocr_result[0]
                cluster_1 = ocr_result[1]
            else:
                cluster_0 = []
                cluster_1 = []

            if len(cluster_0) > 0:
                inside_digit_start_location = scale_location[0]
                inside_digit_end_location = scale_location[-1]
                inside_digit_start_value = meter_config['scale_min_inside']
                inside_digit_end_value = meter_config['scale_max_inside']
                for digit in cluster_0:
                    if inside_digit_start_location <= digit[0] < pointer_location \
                            and digit[1] >= inside_digit_start_value:
                        inside_digit_start_location = digit[0]
                        inside_digit_start_value = digit[1]
                    if pointer_location < digit[0] <= inside_digit_end_location\
                            and digit[1] <= inside_digit_end_value:
                        inside_digit_end_location = digit[0]
                        inside_digit_end_value = digit[1]

                ocr_inside_ratio = (pointer_location - inside_digit_start_location) / (
                        inside_digit_end_location - inside_digit_start_location + 1e-05)
                ocr_inside_digit = inside_digit_start_value + \
                                   (inside_digit_end_value - inside_digit_start_value) * ocr_inside_ratio
            else:
                print('inside digit not found')
                ocr_inside_ratio = (pointer_location - scale_location[0]) / (
                        scale_location[-1] - scale_location[0] + 1e-05)
                ocr_inside_digit = ocr_inside_ratio * (meter_config['scale_max_inside'] - meter_config['scale_min_inside'])

            if len(cluster_1) > 0:
                outside_digit_start_location = scale_location[0]
                outside_digit_end_location = scale_location[-1]
                outside_digit_start_value = meter_config['scale_min_inside']
                outside_digit_end_value = meter_config['scale_max_inside']
                for digit in cluster_1:
                    if outside_digit_start_location <= digit[0] < pointer_location \
                            and digit[1] >= outside_digit_start_value:
                        outside_digit_start_location = digit[0]
                        outside_digit_start_value = digit[1]
                    if pointer_location < digit[0] <= outside_digit_end_location \
                            and digit[1] <= outside_digit_end_value:
                        outside_digit_end_location = digit[0]
                        outside_digit_end_value = digit[1]
                ocr_outside_ratio = (pointer_location - outside_digit_start_location) / (
                        outside_digit_end_location - outside_digit_start_location + 1e-05)
                ocr_outside_digit = outside_digit_start_value + \
                                    (outside_digit_end_value - outside_digit_start_value) * ocr_outside_ratio
            else:
                ocr_outside_ratio = (pointer_location - scale_location[0]) / (
                        scale_location[-1] - scale_location[0] + 1e-05)
                ocr_outside_digit = ocr_outside_ratio * (meter_config['scale_max_outside'] - meter_config['scale_min_outside'])

            result = {'ocr_inside_ratio': ocr_inside_ratio,
                      'ocr_inside_digit': ocr_inside_digit,
                      'ocr_outside_ratio': ocr_outside_ratio,
                      'ocr_outside_digit': ocr_outside_digit}
    return result


def read_process(label_maps, meter_config, meter_center, meter_radius, ocr_result):
    """Get the pointer location relative to the scales.

    Args:
        label_maps (np.array): the label map output from a segment for a meter.
        meter_config (dict): meter_config dict
        ocr_result: (dict) digit location and value
        meter_center: the relative meter center in resized image
        meter_radius: the relative meter radius in resized image
        :
    """
    # Convert the circular meter from Cylindrical(x,y) into Spherical Coordinates(theta,r)
    scale_polar_image = create_polar_image(label_maps, meter_center, meter_radius)
    scale_data, pointer_data = polar_convert_1d_data(scale_polar_image)
    polar_scale_mean_filtration(scale_data)
    result = polar_get_meter_reader(scale_data, pointer_data, ocr_result, meter_config)

    return result


class MeterReader:
    """Find the meters in images and provide a digital readout of each meter.

    Args:
        detector_dir(str): directory of the detector.
        segmenter_dir(str): directory of the segmenter.

    """

    def __init__(self, detector_dir, segmenter_dir):
        if not osp.exists(detector_dir):
            raise Exception("Model path {} does not exist".format(
                detector_dir))
        if not osp.exists(segmenter_dir):
            raise Exception("Model path {} does not exist".format(
                segmenter_dir))
        self.detector = pdx.load_model(detector_dir)
        self.segmenter = pdx.load_model(segmenter_dir)
        # Because we will resize images with (METER_SHAPE, METER_SHAPE) before fed into the segmenter,
        # here the transform is composed of normalization only.
        self.seg_transforms = transforms.Compose([transforms.Normalize()])
        self.ocr = OcrInference()

    def detect_and_crop(self, im_file, meter_config, score_threshold):
        if isinstance(im_file, str):
            im = cv2.imread(im_file).astype('float32')
        else:
            im = im_file.copy()

        # Get detection results
        det_time_start = time.time()
        det_results = self.detector.predict(im)
        det_time_end = time.time()
        print("det time", (det_time_end - det_time_start))
        # Filter bbox whose score is lower than score_threshold
        filtered_results = list()
        for res in det_results:
            if res['score'] > score_threshold:
                filtered_results.append(res)

        for res in filtered_results:
            xmin, ymin, w, h = res['bbox']
            x_center = xmin + w / 2
            y_center = ymin + h / 2
            # read meter config
            if len(meter_config) > 0:
                config_x_center, config_y_center = meter_config[0]['meter_location'][0], \
                                                   meter_config[0]['meter_location'][1]
                dis_min = np.sqrt((x_center - config_x_center) ** 2 + (y_center - config_y_center) ** 2)
                config_ind = 0
                for j in range(len(meter_config)):
                    config_x_center, config_y_center = meter_config[j]['meter_location'][0], \
                                                       meter_config[j]['meter_location'][1]
                    dis = np.sqrt((x_center - config_x_center) ** 2 + (y_center - config_y_center) ** 2)
                    if dis <= dis_min:
                        dis_min = dis
                        config_ind = j
                    MULTI_METER_CONFIG.append(meter_config[config_ind])
        print('meter configuration is done\n')
        resized_meters = list()
        resized_meters_location = list()
        ocr_results = list()
        meter_center = list()
        meter_radius = list()

        for i, res in enumerate(filtered_results):
            # Crop the bbox area
            xmin, ymin, w, h = res['bbox']
            x_center = int(xmin + (w / 2))
            y_center = int(ymin + (h / 2))
            xmin = max(0, int(xmin - (w / 2) * 0.2))
            ymin = max(0, int(ymin - (h / 2) * 0.2))
            xmax = min(im.shape[1], int(xmin + w + (w / 2) * 0.2))
            ymax = min(im.shape[0], int(ymin + h + (h / 2) * 0.2))
            sub_image = im[ymin:(ymax + 1), xmin:(xmax + 1), :]

            # Resize the image with shape (METER_SHAPE, METER_SHAPE)
            meter_shape = sub_image.shape
            scale_x = float(METER_SHAPE) / float(meter_shape[1])
            scale_y = float(METER_SHAPE) / float(meter_shape[0])

            meter_meter = cv2.resize(
                sub_image,
                None,
                None,
                fx=scale_x,
                fy=scale_y,
                interpolation=cv2.INTER_LINEAR)
            circle = find_meter_center(meter_meter.astype(np.uint8))
            meter_center.append([circle[0][0][0], circle[0][0][1]])
            meter_radius.append(circle[0][0][2])
            # ocr process
            # ocr_digit_dict = dict()
            ocr_time_start = time.time()
            ocr_digit_dict = self.ocr.inference(meter_meter)
            if debug:
                print('ocr_digit_dict', ocr_digit_dict)
                print('num of ocr digit', len(ocr_digit_dict))
            ocr_time_end = time.time()
            print("ocr predicting time", (ocr_time_end - ocr_time_start))
            if len(ocr_digit_dict) > 0:
                ocr_img = np.zeros((METER_SHAPE, METER_SHAPE), dtype=np.uint8)
                for key in ocr_digit_dict.keys():
                    location = ocr_digit_dict[key]
                    ocr_img[location[1], location[0]] = 3

                if MULTI_METER_CONFIG[i]['type'] == 'double':
                    if len(ocr_digit_dict) >= 2:
                        dis = distance_from_center_to_ocr(meter_center[i], ocr_digit_dict)
                        # one than one digit
                        cluster_0, cluster_1 = clustering_digits(dis, ocr_digit_dict)
                        # if debug:
                        #     print('cluster_0', cluster_0)
                        #     print('cluster_1', cluster_1)
                        cluster_0 = ocr_check(cluster_0, MULTI_METER_CONFIG[i]['scale_max_inside'])
                        cluster_1 = ocr_check(cluster_1, MULTI_METER_CONFIG[i]['scale_max_outside'])
                        # if debug:
                        #     print('cluster_0_checked', cluster_0)
                        #     print('cluster_1_checked', cluster_1)
                        cluster_0_data = ocr_convert(ocr_img, cluster_0, MULTI_METER_CONFIG[i], circle)
                        cluster_1_data = ocr_convert(ocr_img, cluster_1, MULTI_METER_CONFIG[i], circle)
                        ocr_results.append([cluster_0_data, cluster_1_data])
                        if debug:
                            print('ocr_results', ocr_results)
                    else:
                        # only one ocr digit
                        ocr_digit_data = ocr_convert(ocr_img, ocr_digit_dict, MULTI_METER_CONFIG[i], circle)
                        ocr_results.append([ocr_digit_data])
                else:
                    # single
                    # ocr_digit_dict = ocr_check(ocr_digit_dict, MULTI_METER_CONFIG[i]['scale_max'])
                    ocr_digit_data = ocr_convert(ocr_img, ocr_digit_dict, MULTI_METER_CONFIG[i], circle)
                    ocr_results.append(ocr_digit_data)
                    if debug:
                        print('ocr_results', ocr_results)
            else:
                ocr_results.append([])

            meter_meter = meter_meter.astype('float32')
            resized_meters_location.append((x_center, y_center))
            resized_meters.append(meter_meter)

        return resized_meters_location, resized_meters, meter_center, \
               meter_radius, filtered_results, ocr_results

    def predict(self,
                resized_meters_location,
                resized_meters,
                ocr_results,
                meter_center,
                meter_radius,
                use_erode=True,
                erode_kernel=4,
                seg_batch_size=2):
        """Detect meters in a image, segment scales and points in these meters, the postprocess are
        done to provide a digital readout according to scale and point location.

        Args:
            resized_meters_location (list): for config
            resized_meters (list): for seg mask predict
            ocr_results: ocr digit
            meter_center (list): detected from cropped img
            meter_radius (list):  detected from cropped img
            use_erode (bool, optional): whether to do image erosion by using a specific structuring element for
                the label map output from the segmenter. Default: True.
            erode_kernel (int, optional): structuring element used for erosion. Default: 4.
            seg_batch_size (int, optional): batch size of meters when do segmentation. Default: 2.
        """
        meter_num = len(resized_meters)
        seg_results = list()
        for i in range(0, meter_num, seg_batch_size):
            im_size = min(meter_num, i + seg_batch_size)
            meter_images = list()
            for j in range(i, im_size):
                meter_images.append(resized_meters[j])
            # Segment scales and point in each meter area
            seg_time_start = time.time()
            result = self.segmenter.batch_predict(
                transforms=self.seg_transforms, img_file_list=meter_images)
            seg_time_end = time.time()
            print("seg time", (seg_time_end - seg_time_start))
            if debug:
                for res in result:
                    seg_img = (res['label_map'] * 100)
                    cv2.imshow('before_erode_seg_img', seg_img)
                cv2.waitKey(0)

            # Do image erosion for the predicted label map of each meter
            if use_erode:
                kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
                for k in range(len(result)):
                    result[k]['label_map'] = cv2.erode(result[k]['label_map'], kernel)

            seg_results.extend(result)

            if debug and use_erode:
                for seg_res in seg_results:
                    seg_img_erode = seg_res['label_map'] * 100
                    cv2.imshow('after_erode_seg_img', seg_img_erode)
                    cv2.waitKey(0)

        # The post process are done to get the point location relative to the scales
        results = list()

        for i, seg_result in enumerate(seg_results):
            result = read_process(seg_result['label_map'], MULTI_METER_CONFIG[i],
                                  meter_center[i], meter_radius[i], ocr_results[i])

            results.append(result)
        if debug:
            cv2.destroyAllWindows()

        # Provide a digital readout according to point location relative to the scales
        meter_values = list()
        for i, result in enumerate(results):
            # read meter information from config
            x1, y1 = resized_meters_location[i][0], resized_meters_location[i][1]
            x2, y2 = MULTI_METER_CONFIG[0]['meter_location'][0], MULTI_METER_CONFIG[0]['meter_location'][1]
            dis_min = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            config_index = 0
            if len(MULTI_METER_CONFIG) > 1:
                for j in range(1, len(MULTI_METER_CONFIG)):
                    x2, y2 = MULTI_METER_CONFIG[j]['meter_location'][0], \
                             MULTI_METER_CONFIG[j]['meter_location'][1]
                    dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if dis < dis_min:
                        dis_min = dis
                        config_index = j
            # readout the number form scale value, or pointer's relative position between range
            # or pointer's relative position between ocr digit.
            if result:
                if MULTI_METER_CONFIG[config_index]['type'] == 'single':
                    single_digit_ocr = result['single_digit_ocr']
                    scales = result['scales']
                    print("-- Meter {} -- single_digit_ocr: {} --\n".format(i, single_digit_ocr))
                    print("-- Meter {} -- scales: {} --\n".format(i, scales))
                    if single_digit_ocr:
                        meter_values.append(single_digit_ocr)
                    else:
                        meter_values.append(scales)

                else:
                    ocr_inside_digit = result['ocr_inside_digit']
                    ocr_outside_digit = result['ocr_outside_digit']
                    print("-- Meter {} -- ocr_outside_digit: {} --\n".format(i, ocr_outside_digit))
                    print("-- Meter {} -- ocr_inside_digit: {} --\n".format(i, ocr_inside_digit))
                    if ocr_outside_digit:
                        meter_values.append(ocr_outside_digit)
                    elif ocr_inside_digit:
                        meter_values.append(ocr_inside_digit)
            else:
                meter_values.append(-1)

        return meter_values
