import os

import cv2

from cv_bridge import CvBridge

os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ":/opt/x/lib/cmake/x"
import x_bind as x
import rosbag
import numpy as np

vio = x.VIO()
p = vio.loadParamsFromYaml(
    "params.yaml")
vio.setUp(params=p)
cam_type = p.camera()
tracker_vio = x.Tracker()
tracker_vio.set_params(cam=cam_type, fast_detection_delta=p.fast_detection_delta(),
                       non_max_supp=p.non_max_supp(), block_half_length=p.block_half_length(),
                       margin=p.margin(),
                       n_feat_min=p.n_feat_min(), outlier_method=p.outlier_method(),
                       outlier_param1=p.outlier_param1(), outlier_param2=p.outlier_param2(),
                       win_size_w=p.win_size_w(), win_size_h=p.win_size_h(), max_level=p.max_level(),
                       min_eig_thr=p.min_eig_thr())

bridge = CvBridge()
bag = rosbag.Bag('/home/viciopoli/RPG/bags/circle_high_vel_restamped.bag')
img_count = 0
for topic, msg, _ in bag.read_messages(topics=['/camera/image_raw', '/fcu/imu']):
    t = msg.header.stamp.to_sec()
    if not vio.isInitialized():
        vio.initAtTime(time=t)
    s = None

    if topic == "/fcu/imu":
        w_m = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        a_m = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        s = vio.processImu(timestamp=t, seq=msg.header.seq, w_m=w_m, a_m=a_m)


    if topic == "/camera/image_raw":
        cv_image_gray = bridge.imgmsg_to_cv2(msg, "rgb8")
        cv2.imshow("Frame", cv_image_gray)
        cv2.waitKey(1)
        tracker_vio.track(current_img=cv_image_gray, seq=msg.header.seq, timestamp=t, frame_number=img_count,
                          n_tiles_h=3,
                          n_tiles_w=3, max_feat_per_tile=400)
        matches = tracker_vio.get_matches()
        if matches is not None:
            img = cv2.cvtColor(cv_image_gray, cv2.COLOR_BGR2GRAY)
            s, out = vio.processTracks(timestamp=t, seq=msg.header.seq, matches=matches, image=img)
            cv2.imshow("Frame with tracks", out)
            cv2.waitKey(1)
        img_count += 1

    if not s is None:
        print(f"Postion after feeding: {s.getPosition()}")

cv2.destroyAllWindows()
bag.close()
