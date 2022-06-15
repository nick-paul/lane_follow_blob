#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import cv2 as cv
from dynamic_reconfigure.server import Server
from lane_follow_blob.cfg import BlobConfig
from numpy import ndarray
from lane_follow_blob.vec import Vec
from lane_follow_blob.lane_centering import center_lane
from lane_follow_blob.lane_detection import find_lanes
from lane_follow_blob.utils import rows, cols
import rospkg
import os



class TestImageDemo:

    def __init__(self):
        rospack = rospkg.RosPack()
        self.image_dir = os.path.join(rospack.get_path('lane_follow_blob'), 'test_images')

        self.image = self.load_image(3)
        self.last_debug_image = None
        self.last_image_1 = None

        self.config = None # Dynamic reconfigure
        self.dynamic_reconfigure_server = Server(BlobConfig, self.dynamic_reconfigure_callback)
        

    def load_image(self, image_id: int) -> ndarray:
        image_path = os.path.join(self.image_dir, f'{image_id}.png')
        print(f'Reading image {image_path}')
        return cv.imread(image_path, cv.IMREAD_COLOR)


    def dynamic_reconfigure_callback(self, config, level):
        rospy.logwarn('Got config!')
        self.config = config
        self.process(self.image)
        return config


    def change_image_cb(self, val):
        print(val)
        image = self.load_image(val)
        if image is not None:
            self.image = image
            self.process(self.image)


    def process(self, input_image: ndarray) -> Twist:
        # Create a copy of the input to draw debug data on
        debug_image = input_image.copy()

        # Find the lanes in the image
        lanes_image = find_lanes(input_image, self.config, debug_image=debug_image)
        self.last_image_1 = lanes_image.copy()

        # Run blob lane centering algorithm
        if rospy.get_param('~draw_blob', False):
            p0 = Vec(cols(lanes_image)/2, rows(lanes_image) - rows(lanes_image)/10)
            p_diff = center_lane(lanes_image, p0, debug_image=debug_image, iters=200)
            adjust = p_diff.x
            rospy.loginfo(f'force vector: {p_diff}')

        self.last_debug_image = debug_image
        print('done processing')


rospy.init_node('test_image_demo')
demo = TestImageDemo()
window_setup_done = False
while not rospy.is_shutdown():
    if demo.last_debug_image is not None:
        cv.imshow('final', demo.last_debug_image)
        if not window_setup_done:
            print('assing trackbar')
            cv.createTrackbar('image_id', 'final', 1, 8, demo.change_image_cb)
            window_setup_done = True

    if demo.last_image_1 is not None:
        cv.imshow('med', demo.last_image_1)

    cv.waitKey(1)