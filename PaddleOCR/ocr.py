from paddleocr import PaddleOCR, draw_ocr
import numpy as np
ocr = PaddleOCR(det_model_dir='./Model/ch_ppocr_mobile_v2.0_det_infer',
                rec_model_dir='./Model/ch_ppocr_server_v2.0_rec_infer',
                rec_char_dict_path='/home/zhen/Desktop/MeterProject/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt',
                cls_model_dir = './Model/ch_ppocr_mobile_v2.0_cls_infer',
                use_angle_cls=True,use_gpu=False,use_space_char=True,drop_score=0.7)
img_path = '/home/zhen/Downloads/meter_E57377023_118_2021-04-08T08-37-47.jpg'

def ocr_predict(img_path):

    result = ocr.ocr(img_path)
    number_result = list()
    for line in result:
        Flag = True
        box = line[0]
        score = line[1][1]
        txts = line[1][0]
        for c in txts:
            if c > '9' or c < '0':
                Flag = False
                break
        if Flag:
            number_result.append([box,score,txts])

    return number_result
# 显示结果
# from PIL import Image
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/zhen/github/PaddleOCR/doc/fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')