# MeterReader
The MeterReader system is based on the deep learning, which consists of object detection, semantic segmentation and OCR. It can automatically recognize the meter value.

## Table of Contents

* [Dependence](#1)
* [Usage](#2)
* [The system framework](#3)

## <h2 id="1">Dependence</h2>

The whole system is under PaddlePaddle framework. To getting start, you should go to three repositories below and install the relative dependence. 
1. Install PaddlePaddle

    https://github.com/PaddlePaddle/Paddle
    
2. Install PaddleX

    https://github.com/PaddlePaddle/PaddleX
3. Install PaddleOCR

    https://github.com/PaddlePaddle/PaddleOCR


## <h2 id="2">Usage</h2> 

* Download the code and pretrained Models
```
git clone http://gitlab.devops.intelab.cloud/b.tao/meter_reader.git
```
* Enter project dir
```
cd MeterProject
```
* Predict in the command line
```
python main.py --image_dir='./image_dir'
```
## <h2 id="3">The system framework</h2>

* **Object Detection**: The meters are detected by pretrained object detection model **YoloV3**.
* **Semantic Segmentation**: The mask of pointer and scales was predicted by pretrained segmentation model **DeepLapv3**.
* **OCR**: The digits on the meter are recognized by PaddleOCR
* **Whole Process** :
    1. Crop and resize the images of meters detected by the YoloV3.  
    2. Read the digits from OCR output, and filter digits on the inner side of meter.    
    3. Erode the binary mask image predicted by Deeplabv3 in order to shrink the scales and remove the noise.
    4. Transform the mask image from XY Cartesian Coordinates to Polar Coordinates (rows for degree; cols for radius)
    5. Read the relative positions between pointer, scales and digits. Calculate the value of the meter.      
              

