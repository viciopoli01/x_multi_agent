import os
os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ":/opt/x/lib/cmake/x"

import x_bind as x
import time
import numpy as np

vio = x.VIO()
p = vio.loadParamsFromYaml(
    "/home/viciopoli/RPG/DeNeRaFi/VOND_ws/src/x_multi_agent/samples/ReadBagFile/UAV0_params_thermal.yaml")
vio.setUp(params=p)

t = time.time()
vio.initAtTime(time=t)

t = time.time()
seq = 0
w_m = np.array([0, 1, 2])
a_m = np.array([0, 1, 2])
s = vio.processImu(timestamp=t, seq=seq, w_m=w_m, a_m=a_m)
if s is not None:
    print(f"Postion after IMU feeding: {s.getPosition()}")

t = time.time()
seq = 0
f_1 = x.Feature(timestamp=0.2, frame_number=4, x=5, y=0.1, x_dist=4.1, y_dist=1.1, intensity=0)
f_2 = x.Feature(timestamp=0.2, frame_number=4, x=5, y=0.1, x_dist=4.1, y_dist=1.1, intensity=0)

m = x.Match(previous=f_1, current=f_2)

ml = x.MatchList()
ml.append(m)

s = vio.processTracksNoFrame(timestamp=t, seq=seq, matches=ml, h=100, w=100)
if s is not None:
    print(f"Postion after matches feeding: {s.getPosition()}")
