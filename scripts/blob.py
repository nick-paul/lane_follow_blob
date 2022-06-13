import rospy
from lane_follow_blob import LaneFollowBlob

rospy.init_node('blob')
blob = LaneFollowBlob()
blob.spin()