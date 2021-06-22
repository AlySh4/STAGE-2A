from scipy.spatial.transform import Rotation as R
import numpy as np
from UsefulFunction import tail

vect = [1, 1, 1]
quat = np.array(tail('Data/OptiTrackData.csv', 1)[0][4:8])
print(quat)
rotation = R.from_quat([quat[3], quat[0], quat[1], quat[2]])
print(rotation)
RepereSP = rotation.apply(vect)
print(tuple(RepereSP))

ok = np.array(tail('Data/OptiTrackData.csv', 1)[0][1:4])

print(ok)

#
# RepereSPini = np.array([[1,0,0],
#                      [0,1,0],
#                      [0,0,1]])
# x=1
# x=2
# x=3
#
# with open("Data/OptiTrackData.csv", 'r') as f:
#     line = f.readline()
#     data = list(map(eval,line.strip().split(',')))
#     print(data[4:8])
#     #rotation = R.from_quat(data[4:8])
#     rotation = R.from_quat([0,0,0,1])
#     RepereSP = rotation.apply(RepereSPini)
#     np.diag(RepereSP[1,1] + X, RepereSP[2.])
#     a=
#     print(a)
#     print(RepereSP)
#
