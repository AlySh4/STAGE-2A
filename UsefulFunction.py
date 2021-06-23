import numpy as np
from scipy.spatial.transform import Rotation as R


def tail(fn, n):
    with open(fn, 'r') as f:
        lines = f.readlines()
    return [list(map(eval, line.strip().split(','))) for line in lines[-n:]]


def SmartProbeVect():
    vect = [-0.5, 0, 0]  # vecteur dans le repère mouvant
    quat = np.array(tail('Data/OptiTrackData.csv', 1)[0][4:8])
    rotation = R.from_quat([quat[3], quat[0], quat[1], quat[2]])
    return rotation.apply(vect)


def WindVect():
    angles = np.array(tail('Data/SmartProbeData.csv', 1)[0][2:])
    quat = np.array(tail('Data/OptiTrackData.csv', 1)[0][4:8])
    vect = [-np.cos(angles[0]), -np.sin(angles[1]), -np.sin(angles[0]), ]  # vecteur du vent dans le repère mouvant
    rotation = R.from_quat([quat[3], quat[0], quat[1], quat[2]])
    return rotation.apply(vect)
