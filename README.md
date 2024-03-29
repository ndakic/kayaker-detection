# Detection and Segmentation of Kayakers using Mask R-CNN

This project is based on [Mask R-CNN](https://github.com/matterport/Mask_RCNN) implementation. \
The model generates bounding boxes and segmentation masks for each instance of a kayaker in the image. \
Model is trained on custom dataset and it's based on Feature Pyramid Network (FPN) and a ResNet101 backbone. \
The custom dataset consists of 65 kayaker images with pixel-by-pixel polygon annotations.

Input            |  Output
:-------------------------:|:-------------------------:
![input_video](files/gif/input-video.gif)  |  ![output_video](files/gif/output-video-2.gif)


## Performance measure of the model
Model |  Starting weights | Training Layers | Epochs | Training mAP | Test mAP 
:-------------------------:|:-------------------------:|:---------------:|:------:|:------------:|:-------------------------:
M1 | COCO |      heads      |   5    |    0.844     | 0.800
M2 | COCO |      heads      |   10   |    0.824     | 0.867
M3 | COCO |      heads      |   15   |    0.844     | 0.867
M4 | COCO |      heads      |   20   |    0.844     | 0.800
M5 | COCO |       all       |   5    |    0.865     | 0.867
M6 | COCO |       all       |   10   |    0.885     | 0.933
M7 | COCO |       all       |   15   |    0.865     | 0.867
M8 | COCO |       all       |   20   |    0.885     | 0.933


# Getting Started

### Prerequisites
- Python 3.7

## How To Run Project

1. Clone this repository
 ```git
  git clone https://github.com/ndakic/kayaker-detection
  ```
2. Download yolov3 weights (237 MB) from <a href="https://pjreddie.com/media/files/yolov3.weights">here</a> and add it to your <a href="/model">model folder</a>.
3. Install the requirements using pip and venv 
  ```shell
  pip install -r requirements.txt
  ```
4. Clone and install the Mask R-CNN Library
```
git clone https://github.com/matterport/Mask_RCNN
```
```bash
cd Mask_RCNN
python setup.py install
```
5. Run the following command in the command line
 ```python
  python application.py
  ```
