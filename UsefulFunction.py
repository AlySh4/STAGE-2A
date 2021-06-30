from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor


class TrackClass:
    def __init__(self, length=1000):
        self.length = length
        self.list = deque()

    def TrackStorage(self, position):
        if len(self.list) < self.length:
            self.list.append(position)
        else:
            self.list.popleft()
            self.list.append(position)


class VisuClass:
    def __init__(self, resolution=21, cote=10):
        self.resolution = resolution
        self.cote = cote
        self.matrice = [[[0] * self.resolution] * self.resolution] * self.resolution
        self.X = np.linspace(-cote / 2, cote / 2, self.resolution)
        self.Y = np.linspace(-cote / 2, cote / 2, self.resolution)
        self.Z = np.linspace(0, cote, self.resolution)
        self.Points = self.ListePoints()

    def ListePoints(self):
        P = []
        for x in self.X:
            for y in self.Y:
                for z in self.Z:
                    P.append((x, y, z))
        return np.array(P)


class GPRClass:
    def __init__(self, l=0.1, sigma_f=1):
        self.l = l
        self.sigma_f = sigma_f
        self.kernelx = ConstantKernel(constant_value=self.sigma_f, constant_value_bounds=(1e-2, 1e2)) \
                       * RBF(length_scale=self.l, length_scale_bounds=(1e-2, 1e2))
        self.kernely = ConstantKernel(constant_value=self.sigma_f, constant_value_bounds=(1e-2, 1e2)) \
                       * RBF(length_scale=self.l, length_scale_bounds=(1e-2, 1e2))
        self.kernelz = ConstantKernel(constant_value=self.sigma_f, constant_value_bounds=(1e-2, 1e2)) \
                       * RBF(length_scale=self.l, length_scale_bounds=(1e-2, 1e2))
        self.kernel = self.kernelx * self.kernely * self.kernelz
        self.gp1 = GaussianProcessRegressor(kernel=self.kernel, alpha=self.sigma_f ** 2, n_restarts_optimizer=10, )
        self.gp2 = GaussianProcessRegressor(kernel=self.kernel, alpha=self.sigma_f ** 2, n_restarts_optimizer=10, )
        self.gp3 = GaussianProcessRegressor(kernel=self.kernel, alpha=self.sigma_f ** 2, n_restarts_optimizer=10, )
        self.Xm = None  # Points sur lequels le GPR va être appliqué
        self.Yx = None  # Les coordonnées du vents pour les points sur lequels on fait le training
        self.Yy = None
        self.Yz = None
        self.Yx_pred = None  # Les coordonnées du vents prédits par le GPR
        self.Yy_pred = None
        self.Yz_pred = None

    def setDataForGPR(self, pos, Yx, Yy, Yz):  # fonction qui permet de récuperer les data pour faire le training
        self.Xm = pos
        self.Yx = Yx
        self.Yy = Yy
        self.Yz = Yz

    def runGPR(self):
        self.gp1.fit(self.Xm, self.Yx)
        self.gp2.fit(self.Xm, self.Yy)
        self.gp3.fit(self.Xm, self.Yz)

    def predictWindGPR(self):
        self.Yx_pred = self.gp1.predict(self.Xm)
        self.Yy_pred = self.gp2.predict(self.Xm)
        self.Yz_pred = self.gp3.predict(self.Xm)
        return self.Yx_pred, self.Yy_pred, self.Yz_pred


def tail(fn, n):
    with open(fn, 'r') as f:
        lines = f.readlines()
    return [list(map(eval, line.strip().split(','))) for line in lines[-n:]]


def SmartProbeVect(quat):
    vect = [0.5, 0, 0]  # vecteur dans le repère mouvant
    # quat = np.array(tail('Data/OptiTrackData.csv', 1)[0][4:8])
    rotation = R.from_quat([quat[3], quat[0], quat[1], quat[2]])
    return rotation.apply(vect)


def WindVect(quat, SPData):
    # pour quat np.array(tail('Data/OptiTrackData.csv', 1)[0][4:8]) et SPData = np.array(tail(
    # 'Data/SmartProbeData.csv', 1)[0]) est necessaire
    vect = [np.cos(SPData[3]), np.sin(SPData[3]), np.sin(SPData[2]), ]  # vecteur du vent dans le repère mouvant
    vect = (vect / np.linalg.norm(vect)) * (SPData[0] / 10)
    rotation = R.from_quat([quat[3], quat[0], quat[1], quat[2]])
    return rotation.apply(vect)


def alphaBetaGlobal(v):
    pass
    # TODO: effectuer cela demain.