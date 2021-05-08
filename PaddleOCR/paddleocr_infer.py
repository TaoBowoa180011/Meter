import os
from collections import Counter

import cv2
import numpy as np
from dynaconf import settings

from paddleocr import PaddleOCR, draw_ocr
from PaddleOCR.utility import rotate_image, calculate_iou, order_points


class OcrInference:
    def __init__(self):
        dynaconf_config_ocr = settings.get('ocr', {})
        det_model_dir = os.getcwd() + dynaconf_config_ocr.det_model_dir
        rec_model_dir = os.getcwd() + dynaconf_config_ocr.rec_model_dir
        rec_char_dict_path = os.getcwd() + dynaconf_config_ocr.rec_char_dict_path
        cls_model_dir = os.getcwd() + dynaconf_config_ocr.cls_model_dir
        self.min_angle = dynaconf_config_ocr.min_angle
        self.max_angle = dynaconf_config_ocr.max_angle
        self.angle_step = dynaconf_config_ocr.angle_step
        self.text_score_thresh = dynaconf_config_ocr.text_score_thresh
        self.ocr = PaddleOCR(det_model_dir=det_model_dir,
                             rec_model_dir=rec_model_dir,
                             rec_char_dict_path=rec_char_dict_path,
                             cls_model_dir=cls_model_dir,
                             use_angle_cls=dynaconf_config_ocr.angle_switch,
                             use_gpu=False,
                             )

    def inference(self, image):
        ocr_results = []

        # calculate ocr results with different angle
        for angle in range(self.min_angle, self.max_angle, self.angle_step):

            rotate_img, reversed_rotation_mat = rotate_image(image, angle)

            result = self.ocr.ocr(rotate_img, cls=True)
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]

            res = []
            for i in range(0, len(boxes)):
                score = scores[i]
                box = boxes[i]

                if score <= self.text_score_thresh or not str(txts[i]).isdigit():
                    continue
                ones = np.ones(shape=(len(box), 1))

                points_ones = np.hstack([box, ones])

                # transform points
                transformed_points = reversed_rotation_mat.dot(points_ones.T).T
                convert_int = []
                for points in transformed_points:
                    convert_int.append([float(int(p)) for p in points])
                transformed_points = convert_int
                res.append([[transformed_points], [txts[i], score]])

            result = res

            ocr_results.append(result)
        ocr_cas = []

        # merge results to one list
        for result in ocr_results:
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            for i in range(0, len(boxes)):
                ocr_cas.append([boxes[i], txts[i]])

        # group overlap bounding box
        ocr_output_coords = []
        ocr_output_txts = []
        for i_box, i_txt in ocr_cas:
            i_box = i_box[0]

            for j_box, j_txt in ocr_cas:
                j_box = j_box[0]

                iou = calculate_iou(i_box, j_box)
                if iou > 0.5:
                    if len(ocr_output_coords) == 0:
                        ocr_output_coords.append(
                            [i_box, j_box]
                        )
                        ocr_output_txts.append(
                            [i_txt, j_txt]
                        )

                    else:
                        exists_flag = False
                        for idx, exists_boxes in enumerate(ocr_output_coords):
                            exists_box = exists_boxes[0]
                            iou_exist = calculate_iou(exists_box, j_box)
                            exists_txts = ocr_output_txts[idx]
                            if iou_exist > 0:
                                exists_boxes.append(i_box)
                                exists_txts.append(j_txt)
                                ocr_output_coords[idx] = exists_boxes
                                ocr_output_txts[idx] = exists_txts
                                exists_flag = True
                        if not exists_flag:
                            ocr_output_coords.append(
                                [j_box]
                            )
                            ocr_output_txts.append(
                                [j_txt]
                            )

        output = dict()
        # calculate mean of overlapped bounding box and vote best ocr result
        for locations, texts in zip(ocr_output_coords, ocr_output_txts):
            c = Counter(texts)
            ocr_value, _ = c.most_common()[0]
            location = np.average(locations, axis=0)
            location = [[int(x) for x in lst] for lst in location]

            center = np.sum(location, axis=0)
            center = center / 4
            center = [int(c) for c in center]
            output[ocr_value] = center

        return output


if __name__ == '__main__':
    img = cv2.imread('/home/jy/Projects/meter_reader/paddle_ocr/meter_image/2021-04-09_11-32.jpg')
    ocri = OcrInference()
    print(ocri.inference(img))
