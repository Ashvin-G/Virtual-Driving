# Virtual-Driving

### Prerequisites
This project was developed around Need For Speed: Most Wanted (2012) environment and implemented using Python 3.7.0 and several other external opensource library such as
1. opencv
2. opencv-contrib
3. numpy
4. pyautogui
You can install these packages by following command.
```
$ pip install opencv-python
$ pip install opencv-contrib-python
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

Refer this video [Build and Install OpenCV With CUDA (GPU) Support on Windows 10](https://youtu.be/tjXkW0-4gME) on youtube to install and build OpenCV with GPU support.


### Acknowledgement
The project is inpired from Tanay Karve [Driving using motion recognition](https://github.com/TanayKarve/Driving-using-motion-recogniton) and Sentdex [Python Plays GTA V](https://www.youtube.com/watch?v=ks4MPfMq8aQ&list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a)
