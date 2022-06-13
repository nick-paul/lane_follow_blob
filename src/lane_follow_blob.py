import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class LaneFollowBlob:

    def __init__(self):
        self.cvbridge = CvBridge()

        self.twist_pub = rospy.Publisher('cmd', Twist, queue_size=10)
        rospy.Subscriber('/car/camera1/image_raw', Image, self.input_callback)

    def spin(self):
        rospy.logwarn('Lane follow blob running!!')
        rospy.spin()

    def input_callback(self, msg: Image):
        rospy.loginfo('Got image')
        image = self.cvbridge.imgmsg_to_cv2(msg)
        twist = self.process(image)
        self.twist_pub.publish(twist)


    def process(self, image) -> Twist:
        return Twist