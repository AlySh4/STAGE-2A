from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor


# x1, x2, y1, y2, z1, z2 = -3, -3, -4, -7, 0, 2


class TrackClass:
    def __init__(self, length=1000):
        self.length = length
        self.recentTrack = deque()  # le sillage uniquement
        self.wholeTrack = np.empty((0, 3))  # tout depuis le début de la simulation
        self.wholeWindTrack = np.empty((0, 3))

    def TrackStorage(self, position, Wind):
        if len(self.recentTrack) < self.length:
            self.recentTrack.append(position)
        else:
            self.recentTrack.popleft()
            self.recentTrack.append(position)
        self.wholeTrack = np.concatenate((self.wholeTrack, position.reshape(1, 3)))
        self.wholeWindTrack = np.concatenate((self.wholeWindTrack, (np.array(Wind)).reshape(1, 3)))


class VisuClass:
    def __init__(self, resolution=10, cote=1):
        self.resolution = resolution
        self.cote = cote
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

    def windDistribution(self, X, Y, Z):
        A = np.array((X, Y, Z)).T
        return A + self.Points


class GPRClass:  # TODO: Verifier la fonction de covariance
    def __init__(self, l=0.1, sigma_f=1, espace=None):
        self.l = l
        self.sigma_f = sigma_f
        self.kernelx = ConstantKernel(constant_value=self.sigma_f, constant_value_bounds=(1e-2, 1e2)) \
                       * RBF(length_scale=self.l, length_scale_bounds=(1e-2, 1e2))
        self.kernel = self.kernelx
        self.gp1 = GaussianProcessRegressor(kernel=self.kernel, alpha=self.sigma_f ** 2, n_restarts_optimizer=3, )
        self.gp2 = GaussianProcessRegressor(kernel=self.kernel, alpha=self.sigma_f ** 2, n_restarts_optimizer=3, )
        self.gp3 = GaussianProcessRegressor(kernel=self.kernel, alpha=self.sigma_f ** 2, n_restarts_optimizer=3, )
        self.Xm = None  # Points sur lequels le GPR va être appliqué
        self.Yx = None  # Les coordonnées du vents pour les points sur lequels on fait le training
        self.Yy = None
        self.Yz = None
        self.Yx_pred = None  # Les coordonnées du vents prédits par le GPR
        self.Yy_pred = None
        self.Yz_pred = None
        self.espace = espace

    def setDataForGPR(self, pos, wind):  # fonction qui permet de récuperer les data pour faire le training

        if len(pos) > 200 and len(wind) > 200:
            self.Xm = pos[-300:]
            self.Yx = wind[:, 0][-300:]
            self.Yy = wind[:, 1][-300:]
            self.Yz = wind[:, 2][-300:]
        else:
            self.Xm = pos
            self.Yx = wind[:, 0]
            self.Yy = wind[:, 1]
            self.Yz = wind[:, 2]

    def predictWindGPR(self):
        self.gp1.fit(self.Xm, self.Yx)
        self.gp2.fit(self.Xm, self.Yy)
        self.gp3.fit(self.Xm, self.Yz)
        self.Yx_pred = self.gp1.predict(self.espace)
        self.Yy_pred = self.gp2.predict(self.espace)
        self.Yz_pred = self.gp3.predict(self.espace)


def tail(fn, n):
    with open(fn, 'r') as f:
        lines = f.readlines()
    return [list(map(eval, line.strip().split(','))) for line in lines[-n:]]


def SmartProbeVect(quat):
    vect = [0.5, 0, 0]  # vecteur dans le repère mouvant
    rotation = R.from_quat([quat[3], quat[0], quat[1], quat[2]])
    return rotation.apply(vect)


def WindVect(quat, SPData):
    # pour quat np.array(tail('Data/OptiTrackData.csv', 1)[0][4:8]) et SPData = np.array(tail(
    # 'Data/SmartProbeData.csv', 1)[0]) est necessaire
    vect = [np.cos(SPData[3]), np.sin(SPData[3]), np.sin(SPData[2]), ]  # vecteur du vent dans le repère mouvant
    vect = (vect / np.linalg.norm(vect)) * (SPData[0] / 10)
    rotation = R.from_quat([quat[3], quat[0], quat[1], quat[2]])
    return rotation.apply(vect)
