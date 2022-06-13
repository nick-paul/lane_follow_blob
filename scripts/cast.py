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


def point_on_line(x0, y0, theta, r):
    """
    Use the parametric equation of a line to find a point
    that is r units away from (x0,y0) at theta degrees
    Convert to int t
    """
    x = x0 + r * math.cos(theta)
    y = y0 + r * math.sin(theta)
    return Vec(x, y)


def raycast_find_nonzero(image: ndarray, p0: Vec, theta: float, r_step=5, iters=100, do_draw_points=False) -> List[Vec]:
    """
    Cast a ray from p0 in the direction of theta
    Stop at the first non-zero pixel
    Assumes image is a 3-channel ndarray. If single channel, modify the zero check below
    """
    # Don't start at zero if using draw_point in loop, see note below
    for r in range(3, iters+3):
        p = point_on_line(p0.x, p0.y, theta, -r * r_step)
        p = Vec(int(p.x), int(p.y)) # convert to int since we are indexing pixels

        # Edge of image
        if not (0 < p.x < cols(image) and 0 < p.y < rows(image)):
            # out of bounds, stop
            break

        # Found non-zero
        # Assumes multi-channel image. If single channel, remove channel index `[0]`
        if image[int(p.y), int(p.x)][0] > 0:
            break

        # If drawing points, raycasts after this one may collide with drawn
        #  points, especially near p0. To fix for demo purposes we
        #  use a range above that starts 3 units from p0
        if do_draw_points:
            draw_point(image, p, r=1)

    return p


def compute_spring_force(thetas: ndarray, spring_lengths: ndarray):
    k = 1 # spring constant

    # Use -k so springs pull
    force = -k * spring_lengths

    # Compute force vector along each angle
    force_vectors = [Vec(math.cos(theta), math.sin(theta)).smul(f) for theta, f in zip(thetas, force)]

    # Compute the mean force
    p_diff = Vec(0,0)
    for v in force_vectors:
        p_diff = p_diff.add(v)
    p_diff = p_diff.sdiv(len(force_vectors))

    return p_diff
    

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


    def loop(self):
        image = np.zeros((350,400,3), dtype=np.uint8)
        cfg = self.config

        # Add some test lane lines to the image
        cv.line(image, (cfg.l1_x1,cfg.l1_y1), (cfg.l1_x2, cfg.l1_y2), (10,200,200), 7)
        cv.line(image, (cfg.l2_x1,cfg.l2_y1), (cfg.l2_x2, cfg.l2_y2), (10,200,200), 7)

        # Starting point (p0): Lower center of image
        p0 = Vec(cols(image)/2, rows(image) - rows(image)/10)
        draw_point(image, p0, color=(0,0,255), r=5)

        # The angles of our springs
        thetas = deg2rad(np.array(list(range(0, 180+1, 15))))

        # The locations where the springs intersect the lane lines
        points = [raycast_find_nonzero(image, p0, theta, do_draw_points=True) for theta in thetas]
        for p in points: draw_point(image, p, r=3, color=(255, 50, 50))

        # Compute the overall spring force on the point
        spring_lengths = np.array([Vec.dist(p0, p) for p in points])
        p_diff = compute_spring_force(thetas, spring_lengths)

        # Draw the force vector as a point relative to p0
        p_final = Vec(p0.x + p_diff.x, p0.y + p_diff.y)
        draw_point(image, Vec(p_final.x, p_final.y), color=(0,255,255), r=5)
        
        print('final force vector:', p_diff)

        # Publish debug image
        self.publish_image(image)
        

rospy.init_node('cast')
cast = CastExample()
limiter = rospy.Rate(10)
while not rospy.is_shutdown():
    cast.loop()
    limiter.sleep()