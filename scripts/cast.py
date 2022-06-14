#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge
from numpy import ndarray
import numpy as np
import cv2 as cv
from sensor_msgs.msg import Image
import math
from dynamic_reconfigure.server import Server
from lane_follow_blob.cfg import CastConfig
from typing import List
from lane_follow_blob.vec import Vec
from lane_follow_blob.utils import rows, cols, draw_point, deg2rad
from lane_follow_blob.lane_centering import center_lane


class CastExample:

    def __init__(self):
        self.cvbridge = CvBridge()
        self.cam_pub = rospy.Publisher('/cast_example', Image, queue_size=2)
        self.config = None
        self.dynamic_reconfigure_server = Server(CastConfig, self.dynamic_reconfigure_callback)


    def publish_image(self, image: ndarray):
        self.cam_pub.publish(self.cvbridge.cv2_to_imgmsg(image, encoding='bgr8'))


    def dynamic_reconfigure_callback(self, config, level):
        self.config = config
        return config


    def draw_lanes(self, image):
        cfg = self.config
        cv.line(image, (cfg.l1_x1,cfg.l1_y1), (cfg.l1_x2, cfg.l1_y2), (10,200,200), 7)
        cv.line(image, (cfg.l2_x1,cfg.l2_y1), (cfg.l2_x2, cfg.l2_y2), (10,200,200), 7)



    def loop(self):
        image = np.zeros((350,400,3), dtype=np.uint8)
        debug_image = np.zeros_like(image)
        cfg = self.config

        # Add some test lane lines to the image
        self.draw_lanes(image)
        self.draw_lanes(debug_image)

        # Starting point (p0): Lower center of image
        p0 = Vec(cols(image)/2, rows(image) - rows(image)/10)

        (b, g, r) = cv.split(image)
        p_diff =  center_lane(g, p0, debug_image=debug_image)

        print('final force vector:', p_diff)

        # Publish debug image
        self.publish_image(debug_image)
        

rospy.init_node('cast')
cast = CastExample()
limiter = rospy.Rate(10)
while not rospy.is_shutdown():
    cast.loop()
    limiter.sleep()