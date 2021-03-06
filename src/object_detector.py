#!/usr/bin/env python3
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import argparse
import numpy as np
import time
from threading import Thread
import importlib.util




class object_detector:

  def __init__(self):
    PATH_TO_CKPT = rospy.get_param("/object_detector/weights_path")
    PATH_TO_LABELS=rospy.get_param("/object_detector/labels_path")
    camera_input=rospy.get_param("/object_detector/cam_feed")
    use_tpu=int(rospy.get_param("/object_detector/tpu"))
    self.min_conf_threshold = float(rospy.get_param("/object_detector/threshold"))
    self.imW = int(rospy.get_param("/object_detector/imW"))
    self.imH = int(rospy.get_param("/object_detector/imH"))
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_tpu:
          from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_tpu:
          from tensorflow.lite.python.interpreter import load_delegate
    if use_tpu:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
      if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'  
    with open(PATH_TO_LABELS, 'r') as f:
        self.labels = [line.strip() for line in f.readlines()]
    if self.labels[0] == '???':
        del(self.labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_tpu:
        self.interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        self.interpreter = Interpreter(model_path=PATH_TO_CKPT)
    self.interpreter.allocate_tensors()

    # Get model details
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()
    self.height = self.input_details[0]['shape'][1]
    self.width = self.input_details[0]['shape'][2]

    self.floating_model = (self.input_details[0]['dtype'] == np.float32)

    self.input_mean = 127.5
    self.input_std = 127.5
    # Initialize frame rate calculation
    self.frame_rate_calc = 1
    self.freq = cv2.getTickFrequency()

    self.image_pub = rospy.Publisher("/detected_image",Image,queue_size=10)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(camera_input,Image,self.callback)
  def callback(self,data):
    t1 = cv2.getTickCount()
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    frame = cv_image.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if self.floating_model:
        input_data = (np.float32(input_data) - self.input_mean) / self.input_std

    # Perform the actual detection by running the model with the image as input
    self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
    self.interpreter.invoke()
    # Retrieve detection results
    boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
    scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)


    output=[x for x in zip(classes,boxes,scores) if x[2]>self.min_conf_threshold and x[2]<=1.0]

    print("Output",output)


    # try:
    #   # self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
    # except CvBridgeError as e:
    #   print(e)

def main():
  ic = object_detector()
  rospy.init_node('object_detector', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main()