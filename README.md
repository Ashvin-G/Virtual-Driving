# Virtual-Driving

### Prerequisites
This program was implemented using Python 3.7.0 and several other external opensource library such as
1. OpenCV
2. numpy
3. pyautogui
You can install these packages by following command.
```
$ pip install opencv-python
$ pip install numpy
$ pip install pyautogui
```
The program is capable of detecting cars, truck and train in the video stream. This was capable by using MobileNet-SSD (Single shot detection) network and Caffe framework which is intended to perform object detection. The detection algorithm is amplified by using NVIDIA CUDA support for OpenCV which uses GPU.

If your OpenCV is not built and install with CUDA(GPU) support then replace
```
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```
with
```
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CPU)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_TARGET)
```
in net_config.py file.

