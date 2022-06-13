import rospy
from lane_follow_blob.lane_follow_blob_port import LaneFollowBlob

rospy.init_node('blob')
blob = LaneFollowBlob()
blob.spin()