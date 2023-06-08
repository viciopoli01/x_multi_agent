#!/usr/bin/env python3

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

import x_bind as x

class VIO():
    def __init__(self):
        self.vio = x.VIO()
        p = self.vio.loadParamsFromYaml(
            "/home/viciopoli/RPG/DeNeRaFi/VOND_ws/src/VOND/front_end/params.yaml")
        cp = x.CamParams()
        p.camera_params(cp)
        self.vio.setUp(params=p)
        self.cam_type = x.CameraFov(cp)
        ######
        self.bridge = CvBridge()
        self.W, self.H = 752, 480
        K = np.array([
            [cp.fx(), 0, cp.cx()],
            [0, cp.fy(), cp.cy()],
            [0, 0, 1]
        ])
        # self.W //= 4
        # self.H //= 4
        # K //= 4
        self.tracker_vio = x.Tracker()
        self.tracker_vio.set_params(cam=self.cam_type, fast_detection_delta=p.fast_detection_delta(),
                                    non_max_supp=p.non_max_supp(), block_half_length=p.block_half_length(),
                                    margin=p.margin(),
                                    n_feat_min=p.n_feat_min(), outlier_method=p.outlier_method(),
                                    outlier_param1=p.outlier_param1(), outlier_param2=p.outlier_param2(),
                                    win_size_w=p.win_size_w(), win_size_h=p.win_size_h(), max_level=p.max_level(),
                                    min_eig_thr=p.min_eig_thr())

        rospy.Subscriber("/camera/image_raw", Image, self.img_callback)
        rospy.Subscriber("/fcu/imu", Imu, self.imu_callback)
        # Publishers
        self.pub = rospy.Publisher('~/img_matches', Image, queue_size=10)
        self.pose_pub_imu = rospy.Publisher('~/pose_vio_imu', PoseStamped, queue_size=100)
        self.pose_pub_cam = rospy.Publisher('~/pose_vio_cam', PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher('~/markers', Marker, queue_size=10)

    def img_callback(self, data):
        # Try to convert the ROS Image message to a CV2 Image
        seq = data.header.seq
        t = data.header.stamp.to_sec()
        try:
            cv_image_gray = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.tracker_vio.track(current_img=cv_image_gray, seq=seq, timestamp=t, frame_number=seq, n_tiles_h=3,
                                   n_tiles_w=3, max_feat_per_tile=400)

            matches = self.tracker_vio.get_matches()
            if matches is not None:
                s, out = self.vio.processTracks(timestamp=t, seq=seq, matches=matches, image=cv_image_gray)
                if s is not None:
                    markers = self.vio.computeSLAMCartesianFeaturesForState(s)
                    for i, f in enumerate(markers):
                        self.pub_markers(f=f, t=data.header.stamp, id=i)
                    self.pub_pose(s, self.pose_pub_cam)

                self.pub.publish(self.bridge.cv2_to_imgmsg(out, encoding='rgb8'))
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def imu_callback(self, data):
        t = data.header.stamp.to_sec()
        if not self.vio.isInitialized():
            self.vio.initAtTime(t)
        seq = data.header.seq
        w_m = np.array([data.angular_velocity.x,
                        data.angular_velocity.y,
                        data.angular_velocity.z])
        a_m = np.array([data.linear_acceleration.x,
                        data.linear_acceleration.y,
                        data.linear_acceleration.z])
        s = self.vio.processImu(timestamp=t, seq=seq, w_m=w_m, a_m=a_m)
        if s is not None:
            self.pub_pose(s, self.pose_pub_imu)

    def pub_pose(self, s, pub):
        p = PoseStamped()
        p.header.frame_id = 'world'
        pos = s.getPosition()
        p.pose.position.x = pos[0]
        p.pose.position.y = pos[1]
        p.pose.position.z = pos[2]
        # Make sure the quaternion is valid and normalized
        ori = s.getOrientation()
        p.pose.orientation.x = ori[1]
        p.pose.orientation.y = ori[2]
        p.pose.orientation.z = ori[3]
        p.pose.orientation.w = ori[0]
        pub.publish(p)

    def pub_markers(self, f, t, id):
        point = Marker()

        point.header.frame_id = "world"
        point.header.stamp = t
        point.action = point.ADD
        point.type = point.SPHERE
        point.lifetime = rospy.Duration()
        point.ns = "slam_features"
        point.id = id

        point.scale.x = 0.06
        point.scale.y = 0.06
        point.scale.z = 0.06

        point.color.r = 50.0 / 255.0
        point.color.g = 1.0 / 2.0
        point.color.b = 50.0 / 255.0
        point.color.a = 1.0

        point.pose.position.x = f[0]
        point.pose.position.y = f[1]
        point.pose.position.z = f[2]
        point.pose.orientation.x = 0.0
        point.pose.orientation.y = 0.0
        point.pose.orientation.z = 0.0
        point.pose.orientation.w = 1.0

        self.marker_pub.publish(point)


if __name__ == '__main__':
    # inito ROS node
    rospy.init_node("front_node")

    VIO()

    # Spin ROS node
    rospy.spin()
