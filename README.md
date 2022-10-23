# Mask R-CNN for Kayaker Detection and Segmentation 

This project uses [Mask R-CNN](https://github.com/matterport/Mask_RCNN) implementation for kayaker detection. \
Model (trained on custom dataset) generates bounding boxes and segmentation masks for each instance of a kayaker in the frame. \
It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone. \
The custom dataset consists of 65 kayaker images with pixel-by-pixel polygon annotation.

Input            |  Output
:-------------------------:|:-------------------------:
![input_video](files/gif/input-video.gif)  |  ![output_video](files/gif/output-video-2.gif)


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