# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import cv2
import os
import os.path as osp
from meter_reader.meter import MeterReader
import paddlex as pdx
import time

def visualize(im_file, filtered_results, meter_values, save_dir='./'):
    # Visualize the results
    visual_results = list()
    for i, res in enumerate(filtered_results):
        # Use `score` to represent the meter value
        res['score'] = meter_values[i]
        visual_results.append(res)
    pdx.det.visualize(im_file, visual_results, -1, save_dir=save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Meter Reader Infering')
    parser.add_argument(
        '--detector_dir',
        dest='detector_dir',
        help='The directory of models to do detection',
        default='/home/zhen/Desktop/MeterProject/Model/meter_det_inference_model',
        type=str)
    parser.add_argument(
        '--segmenter_dir',
        dest='segmenter_dir',
        help='The directory of models to do segmentation',
        default='/home/zhen/Downloads/meter_inference_model_new',
        type=str)
    parser.add_argument(
        '--image_dir',
        dest='image_dir',
        help='The directory of images to be infered',
        type=str,
        default='/home/zhen/images/meter/seg/img_beter_crop/')
    # default='/home/zhen/github/PaddleX/examples/meter_reader/meter_test')
    parser.add_argument(
        '--image',
        dest='image',
        help='The image to be infered',
        type=str,
        default='/home/zhen/images/meter/seg/img_beter_crop/meter42021-04-29_13:03:59.png')
    # '/home/zhen/github/PaddleX/examples/meter_reader/meter_test/20190822_142.jpg')
    parser.add_argument(
        '--use_camera',
        dest='use_camera',
        help='Whether use camera or not',
        action='store_true')
    parser.add_argument(
        '--camera_id',
        dest='camera_id',
        type=str,
        help='The camera id',
        default=None)
    #'rtmp://rtmp01open.ys7.com/openlive/ac7cd65f8c2e414184840403a1692238.hd'
    parser.add_argument(
        '--use_erode',
        dest='use_erode',
        help='Whether erode the predicted lable map',
        default=True)
    parser.add_argument(
        '--erode_kernel',
        dest='erode_kernel',
        help='Erode kernel size',
        type=int,
        default=4)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the Model results',
        type=str,
        default='meter_reader/output/result_rectified_3')
    parser.add_argument(
        '--score_threshold',
        dest='score_threshold',
        help="Detected bbox whose score is lower than this threshold is filtered",
        type=float,
        default=0.4)
    parser.add_argument(
        '--seg_batch_size',
        dest='seg_batch_size',
        help="Segmentation batch size",
        type=int,
        default=2)

    return parser.parse_args()





def is_pic(img_name):
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    if suffix not in valid_suffix:
        return False
    return True


def meter_read(args, meter_config_dict):
    image_lists = list()
    if args.image is not None:
        if not osp.exists(args.image):
            raise Exception("Image {} does not exist.".format(args.image))
        if not is_pic(args.image):
            raise Exception("{} is not a picture.".format(args.image))
        image_lists.append(args.image)
    elif args.image_dir is not None:
        if not osp.exists(args.image_dir):
            raise Exception("Directory {} does not exist.".format(
                args.image_dir))
        for im_file in os.listdir(args.image_dir):
            if not is_pic(im_file):
                continue
            im_file = osp.join(args.image_dir, im_file)
            image_lists.append(im_file)

    meter_reader = MeterReader(args.detector_dir, args.segmenter_dir)
    if len(image_lists) > 0:
        for im_file in image_lists:
            print(im_file)
            resized_meters_location, resized_meters, meter_center, meter_radius, filtered_results, ocr_results = \
                meter_reader.detect_and_crop(im_file, meter_config_dict, args.score_threshold)

            meter_values = meter_reader.predict(resized_meters_location, resized_meters, ocr_results,
                                                meter_center, meter_radius, args.use_erode,
                                                args.erode_kernel, args.seg_batch_size)

            visualize(im_file, filtered_results, meter_values, save_dir=args.save_dir)
    elif args.camera_id:
        cap_video = cv2.VideoCapture(args.camera_id)
        if not cap_video.isOpened():
            raise Exception(
                "Error opening video stream, please make sure the camera is working"
            )

        while cap_video.isOpened():
            ret, frame = cap_video.read()
            if ret:
                resized_meters_location, resized_meters, meter_center, meter_radius, filtered_results, ocr_results = \
                    meter_reader.detect_and_crop(frame, meter_config_dict, args.score_threshold)

                meter_values = meter_reader.predict(resized_meters_location, resized_meters, ocr_results,
                                                    meter_center, meter_radius, args.use_erode,
                                                    args.erode_kernel, args.seg_batch_size)

                visualize(frame, filtered_results, meter_values, save_dir=args.save_dir)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap_video.release()


if __name__ == '__main__':
    meter_config = [
        # {
        #     'meter_location': (100, 70),
        #     'scale_value': 0.5,
        #     'scale_min': 0,
        #     'scale_max': 25,
        #     'type': 'single'},
        # {
        #     'meter_location': (50, 70),
        #     'scale_value': 0.02,
        #     'scale_min': 0,
        #     'scale_max': 0.6,
        #     'type': 'single'}
        # ,
        {
            'meter_location': (150, 700),
            'scale_value_inside': 2,
            'scale_value_outside': 0.02,
            'scale_min_inside': 0,
            'scale_max_inside': 80,
            'scale_min_outside': 0,
            'scale_max_outside': 0.6,
            'type': 'double'
        },
        {
            'meter_location': (500, 700),
            'scale_value_inside': 100,
            'scale_value_outside': 1,
            'scale_min_inside': 0,
            'scale_max_inside': 3500,
            'scale_min_outside': 0,
            'scale_max_outside': 25,
            'type': 'double'
        }
    ]
    args = parse_args()
    time_start = time.time()
    meter_read(args, meter_config)
    time_end = time.time()
    time_ = time_end - time_start
    print("time", time_)