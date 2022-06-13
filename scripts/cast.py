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

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return f'({self.x},{self.y})'
    def __repr__(self):
        return f'({self.x},{self.y})'
    def mul(self,s):
        return Point(self.x*s,self.y*s)

def dist(p0, p1) -> float:
    return math.sqrt((p0.x-p1.x)**2 + (p0.y-p1.y)**2)

def rows(mat: ndarray) -> int:
    return mat.shape[0]

def cols(mat: ndarray) -> int:
    return mat.shape[1]

def draw_point(image: ndarray, p: Point, color=(255,255,255), r=5):
    cv.circle(image, (int(p.x), int(p.y)), int(r), color, -1)

def rad(deg):
    return deg * (math.pi / 180)
    
def point_on_line(x0, y0, theta, r):
    x = x0 + r * math.cos(theta)
    y = y0 + r * math.sin(theta)
    return Point(int(x), int(y))

#def spring(p0, points, blob_coeff=0.1, blob_len=0.2) -> Point:
#    p_a = Point(0,0)
#    rospy.loginfo(f'p_a: {p_a}')
#    for p in points:
#        diffx = p0.x - p.x
#        diffy = p0.y - p.y
#
#        p_a.x += p.x
#        p_a.y += p.y
#
#        length = math.sqrt(diffx * diffx + diffy * diffy);
#
#        if length < .01: continue
#
#        diffx /= length
#        diffy /= length
#
#        spring_force = -1 * blob_coeff * (length - blob_len)
#
#        diffx *= spring_force
#        diffy *= spring_force
#
#        p0.x = diffx
#        p0.y = diffy
#
#    p_a.x /= len(points)
#    p_a.y /= len(points)
#
#    p0.x = p_a.x + p0.x
#    p0.y = p_a.y + p0.y
#
#    return p0


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
        print('loop')
        image = np.zeros((350,400,3), dtype=np.uint8)
        cfg = self.config

        # Draw a line on the image
        #cv.line(image, (230,190), (320, 320), (10,200,200), 7)
        #cv.line(image, (190,190), (30, 320), (10,200,200), 7)
        cv.line(image, (cfg.l1_x1,cfg.l1_y1), (cfg.l1_x2, cfg.l1_y2), (10,200,200), 7)
        cv.line(image, (cfg.l2_x1,cfg.l2_y1), (cfg.l2_x2, cfg.l2_y2), (10,200,200), 7)

        # Lower center of image
        p0 = Point(cols(image)/2, rows(image) - rows(image)/10)
        draw_point(image, p0, color=(0,0,255), r=5)

        points = []
        r_step = 5 #math.sqrt(2)
        thetas = rad(np.array(list(range(0, 180+1, 15))))
        for theta in thetas:
            for r in range(3, 100):
                pn = point_on_line(p0.x, p0.y, theta, -r * r_step)

                # Edge of image
                if not (0 < pn.x < cols(image) and 0 < pn.y < rows(image)):
                    # out of bounds, stop
                    points.append(pn)
                    break

                # Found line
                if image[int(pn.y), int(pn.x)][0] > 0:
                    points.append(pn)
                    break

                draw_point(image, pn, r=1)


        for p in points:
            draw_point(image, p, r=3, color=(255, 50, 50))


        # C = cols(image)
        # R = rows(image)
        # p_result = spring(Point(p0.x/C, p0.y/R), [Point(p.x/C, p.y/R) for p in points])
        # rospy.loginfo(p_result)
        # draw_point(image, Point(p_result.x * C, p_result.y * R), color=(0,255), r=5)

        # Compute spring lengths
        spring_lengths = np.array([dist(p0, p) for p in points])

        # Compute spring pull
        k = 1
        force = k * spring_lengths * -1 # -1: pull
        normalized_vectors = [Point(math.cos(theta), math.sin(theta)).mul(f) for theta, f in zip(thetas, force)]
        p_diff = Point(0,0)
        for v in normalized_vectors:
            p_diff.x += v.x
            p_diff.y += v.y
        p_diff.x /= len(normalized_vectors)
        p_diff.y /= len(normalized_vectors)
        
        p_final = Point(p0.x + p_diff.x, p0.y + p_diff.y)
        draw_point(image, Point(p_final.x, p_final.y), color=(0,255,255), r=5)




        self.publish_image(image)
        







rospy.init_node('cast')
cast = CastExample()
limiter = rospy.Rate(10)
while not rospy.is_shutdown():
    cast.loop()
    limiter.sleep()