# Kayaker Detection with Mask R-CNN


### Prerequisites
- Python 3.7


### Result

Input            |  Output
:-------------------------:|:-------------------------:
![input_video](https://github.com/ndakic/kayaker-detection/blob/main/files/input-video.gif)  |  ![output_video](https://github.com/ndakic/kayaker-detection/blob/main/files/output-video.gif)


### How to run

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