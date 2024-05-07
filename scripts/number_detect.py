​#!/usr/bin/env python  
# -*- coding: utf-8 -*  
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import tensorflow as tf
from math import sqrt
from tensorflow.keras.models import load_model
from NumberNet.predict_num import NumModel
​

class RosTensorFlow():
    def __init__(self):
        self._cv_bridge = CvBridge()
​
        #self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        self.model = load_model(os.path.join('/home/nvidia/Downloads/NumberNet-1.1/NumberNet', 'model'))
        self._sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('result', Int16, queue_size=1)
​
    def img_processe(self, img):
        processed = cv2.resize(img, (32, 32))
        processed = cv2.equalizeHist(processed)
        _, processed = cv2.threshold(processed, 40, 255, cv2.THRESH_BINARY)
        return processed
​
    def predict(self, img):
        img = img / 255
        img = img.reshape(1, 32, 32, 1)
        predict = self.model.predict(img)
        classno = np.argmax(predict, axis=1)
        probVal = np.amax(predict)
        # 返回类别和可信度
        return classno, probVal
​
    def get_num_roi(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1.5, 400, param1=100, param2=0.9, minRadius=40, maxRadius=300)
        # 若圆存在,且存在两个圆,且两个圆圆心距不超过10像素点，选中小圆
        if circles is not None:
​
            if circles.shape[0] == 1:
                x = circles[0, 0, 0]
                y = circles[0, 0, 1]
                r = circles[0, 0, 2]
​
            if circles.shape[0] == 2 and (abs(circles[0, 0, 0] - circles[1, 0, 0]) < 10):
                if circles[0, 0, 2] < circles[1, 0, 2]:
                    small_index = 0
                else:
                    small_index = 1
​
                # 小圆圆心坐标，半径
                x = circles[small_index, 0, 0]
                y = circles[small_index, 0, 1]
                r = circles[small_index, 0, 2]
            
            if circles.shape[0] >2:
                return None, None, None
​
            x1 = int(x - r * sqrt(2) / 2)
            y1 = int(y - r * sqrt(2) / 2)
            # ROI右下角坐标
            x2 = int(x + r * sqrt(2) / 2)
            y2 = int(y + r * sqrt(2) / 2)
            # 检查是否越界
            if x1 > 0 and y1 > 0 and x2 < gray.shape[1] and y2 < gray.shape[0]:
                return gray[y1:y2, x1:x2], int(x), int(y)
            else:
                return None, None, None
        # 圆识别失败
        else:
            return None, None, None
​
​
    def callback(self, image_msg):
        img = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        roi, x, y = self.get_num_roi(img)
        if roi is not None:
                processed = self.img_processe(roi)
                classno, prob = self.predict(processed)
                if prob > 0.9:
                      print("结果为:", classno)
                      print("zhixindu结果为:", prob)
                      self._pub.publish(int(classno))
        #img = img / 255
        #img = img.reshape(1, 32, 32, 1)
        #predict = self.model.predict(img)
        #classno = np.argmax(predict, axis=1)
        #probVal = np.amax(predict)
        #print("结果为:", classno)
        #model = NumModel(cv_image)
        #num, pro, x, y = model.detect_show()
        #print("结果为:", num)
        #cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        #ret,cv_image_binary = cv2.threshold(cv_image_gray,128,255,cv2.THRESH_BINARY_INV)
        #cv_image_28 = cv2.resize(cv_image_binary,(28,28))
        #np_image = np.reshape(cv_image_28,(1,28,28,1))
        #predict_num = self._session.run(self.y_conv, feed_dict={self.x:np_image,self.keep_prob:1.0})
        #answer = np.argmax(predict_num,1)
        #rospy.loginfo('%d' % answer)
        #self._pub.publish(answer)
​
    def main(self):
        rospy.spin()
​
if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()
​
​
​