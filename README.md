# Mask R-CNN for Kayaker Detection and Segmentation 


This projects shows a methodological approach with a transfer learning technique for kayakers detection  and instance segmentation using the mask region proposal convolutional neural network (Mask R-CNN). 
Custom dataset used in this project is consist of 65 kayaker images with pixel-by-pixel polygon annotation for the automatic segmentation task. 
The proposed transfer learning technique makes use of a Mask R-CNN model pre-trained on Microsoft Coco dataset.  The pre-trained model is later fine-tuned on custom dataset.

![output_video](https://github.com/ndakic/kayaker-detection/blob/main/files/gif/output-video-2.gif)


# Getting Started

## Prerequisites
- Python 3.7

## How To Run Project

1. Clone this repository
 ```git
  git clone https://github.com/ndakic/kayaker-detection
  ```

2. Download yolov3 weights (237 MB) from <a href="https://pjreddie.com/media/files/yolov3.weights">here</a> and add it to your <a href="/model">model folder</a>.
3. Install the requirements using pip and venv 
  ```python
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