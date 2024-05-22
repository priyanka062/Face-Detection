### FACE DETECTION
##  Introduction:

This face detection program is developed using Python, OpenCV library, and object-oriented programming concepts. The program uses a Haar Cascade classifier and edge detection techniques to detect faces in images or real-time video streams.

Prerequisites:

To run this program, you will need the following software installed on your machine:

- Python 3.x
- OpenCV library
- Numpy library

Haar Cascade Classifier:

A Haar Cascade classifier is a machine learning-based approach for object detection. It uses a set of Haar-like features to detect faces in an image or video stream. The classifier is trained using positive and negative samples of the object to detect. In this program, we use the pre-trained frontal face Haar Cascade classifier provided by OpenCV.

Edge Detection:

Edge detection is a technique used in image processing to identify edges or boundaries in an image. In this program, we use the Canny edge detection algorithm to identify edges in an image. This helps in detecting the edges of the face and separating it from the background.

Object-Oriented Programming:

This program is developed using object-oriented programming concepts. The main classes used in this program are:

- FaceDetector: This class is responsible for detecting faces in an image or video stream using the Haar Cascade classifier and Canny edge detection algorithm.
- VideoStream: This class is responsible for capturing and processing real-time video streams.

##  Usage:

To use this program, you can run the `main.py` file.Sure, here's a brief usage:

Usage:

To detect faces in an image and draw a square around them, call the `face_cascade.detectMultiScale(image_path)` function with the path to the image file as a parameter.
The function will display the image with the face detection squares overlaid. If there are multiple faces in the image, the function will draw a square around each face. If there are no faces in the image, the function will display a message indicating that no faces were detected.

Note that this function uses the pre-trained frontal face Haar Cascade classifier provided by OpenCV for face detection.

##  Conclusion:

This face detection program is an example of how to use Python, OpenCV library, and object-oriented programming concepts to detect faces in images or real-time video streams. The program uses a Haar Cascade classifier and edge detection techniques to achieve high accuracy in face detection.
