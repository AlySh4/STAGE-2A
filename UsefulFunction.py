"""
MIT License

Copyright (c) 2021 Aly SHAHIN <aly.shahin4@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import colorsys
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

import pprzlink.serial
import pprzlink.messages_xml_map as messages_xml_map
import pprzlink.message as message
import time


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
    """
    Class deals with everything relative tot the configuration of
    """

    def __init__(self, d=3, x1=-2, x2=2, y1=0, y2=2, z1=-2, z2=2):
        self.d = d
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        self.resX = int(abs(self.x2 - self.x1) * self.d)
        self.resY = int(abs(self.y2 - self.y1) * self.d)
        self.resZ = int(abs(self.z2 - self.z1) * self.d)
        # print(self.resX, self.resY, self.resZ)
        self.nbpoint = self.resX * self.resY * self.resZ
        self.X = np.linspace(self.x1, self.x2, self.resX)
        self.Y = np.linspace(self.y1, self.y2, self.resY)
        self.Z = np.linspace(self.z1, self.z2, self.resZ)
        self.Points = self.ListePoints()
        self.Xcut = np.insert(np.transpose([np.tile(self.Y, self.resZ), np.repeat(self.Z, self.resY)]), 0,
                              [0] * (self.resY * self.resZ), axis=1)
        self.Ycut = np.insert(np.transpose([np.tile(self.X, self.resZ), np.repeat(self.Z, self.resX)]), 1,
                              [0] * (self.resX * self.resZ), axis=1)
        self.Zcut = np.insert(np.transpose([np.tile(self.X, self.resY), np.repeat(self.Y, self.resX)]), 2,
                              [0] * (self.resX * self.resY), axis=1)
        self.xcuut = 0
        self.ycuut = 0
        self.zcuut = 0
        self.xcuutVariance = 0
        self.ycuutVariance = 0
        self.zcuutVariance = 0

    def ListePoints(self):
        P = []
        for x in self.X:
            for y in self.Y:
                for z in self.Z:
                    P.append((x, y, z))
        return np.array(P)

    def newsetCut(self, component, cut):
        if component == 'x':
            self.xcuut = cut
        if component == 'y':
            self.ycuut = cut
        if component == 'z':
            self.zcuut = cut

    def newsetCutVariance(self, component, cut):
        if component == 'x':
            self.xcuutVariance = cut
        if component == 'y':
            self.ycuutVariance = cut
        if component == 'z':
            self.zcuutVariance = cut

    def windDistribution(self, X, Y, Z):
        A = np.array((X, Y, Z)).T
        return A + self.Points


class GPRClass:
    """
    Class deals woth everything relative to the Gaussian process Regeression Algorithm
    """

    def __init__(self, l=0.1, sigma_f=1, espace=None):
        self.l = l
        self.sigma_f = sigma_f
        self.kernel = ConstantKernel(constant_value=self.sigma_f, constant_value_bounds=(1e-2, 1e2)) \
                      * RBF(length_scale=self.l, length_scale_bounds=(1e-2, 1e2))

        # incertitude
        self.variance = self.kernel
        ###

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
        self.Xcut_pred = None
        self.Ycut_pred = None
        self.Zcut_pred = None
        self.Xcut_predVariance = None
        self.Ycut_predVariance = None
        self.Zcut_predVariance = None
        self.espace = espace
        self.varianceX = None
        self.varianceY = None
        self.varianceZ = None

    def setDataForGPR(self, pos, wind):
        """
        method to get the GPR traing set
        """

        # if len(pos) > 100 and len(wind) > 100:
        #     self.Xm = pos[-100:-1]
        #     self.Yx = wind[:, 0][-100:-1]
        #     self.Yy = wind[:, 1][-100:-1]
        #     self.Yz = wind[:, 2][-100:-1]
        # else:
        self.Xm = pos
        self.Yx = wind[:, 0]
        self.Yy = wind[:, 1]
        self.Yz = wind[:, 2]

    def predictWindGPR(self):
        """
        To predict the GPR and the variance of each component
        """
        self.gp1.fit(self.Xm, self.Yx)
        self.gp2.fit(self.Xm, self.Yy)
        self.gp3.fit(self.Xm, self.Yz)
        self.Yx_pred, covx = self.gp1.predict(self.espace, return_cov=True)
        self.Yy_pred, covy = self.gp2.predict(self.espace, return_cov=True)
        self.Yz_pred, covz = self.gp3.predict(self.espace, return_cov=True)
        self.varianceX = np.diagonal(covx)
        self.varianceY = np.diagonal(covy)
        self.varianceZ = np.diagonal(covz)

    def Variance(self, i):
        """
        Get the max of the three varaince for a point in the studied space.
        """
        return max(self.varianceX[i], self.varianceY[i], self.varianceZ[i])

    def VarianceSecViewGPR(self, component, resX, resY, resZ, Xcut, Ycut, Zcut):
        """
        Take a cut of the choosen space to calculate the incertitude on the secondary view
        """
        if component == 'x' or component == 'all':
            self.Xcut_predVariance = self.varianceX[Xcut * (resY * resZ):(Xcut + 1) * (resY * resZ)]
        if component == 'y' or component == 'all':
            P = np.array([])
            for j in range(resX):
                P = np.concatenate(
                    (P, self.varianceY[Ycut * resZ + j * (resZ * resY):(Ycut + 1) * resZ + j * (resZ * resY)]),
                    axis=None)
            self.Ycut_predVariance = P
        if component == 'z' or component == 'all':
            self.Zcut_predVariance = self.varianceZ[Zcut::resZ]

    def WindSecViewGPR(self, component, resX, resY, resZ, Xcut, Ycut, Zcut):
        """
        Take a cut of the choosen space to calculate the wind projection on the secondary view
        """
        if component == 'x' or component == 'all':
            self.Xcut_pred = self.Yx_pred[Xcut * (resY * resZ):(Xcut + 1) * (resY * resZ)]
        if component == 'y' or component == 'all':
            P = np.array([])
            for j in range(resX):
                P = np.concatenate(
                    (P, self.Yy_pred[Ycut * resZ + j * (resZ * resY):(Ycut + 1) * resZ + j * (resZ * resY)]),
                    axis=None)
            self.Ycut_pred = P
        if component == 'z' or component == 'all':
            self.Zcut_pred = self.Yz_pred[Zcut::resZ]


def tail(fn, n):
    """
    Take the last n line of a fn file into an np.array
    """
    with open(fn, 'r') as f:
        lines = f.readlines()
        # print (lines)
    return [list(map(eval, line.strip().split(','))) for line in lines[-n:]]


def SmartProbeVect(quat):
    """
    Calculate the smartprobe vect from the mooving coordonate sytem to the global coordonate system
    """
    vect = [0, -0.5, 0]  # SP vector in the mooving coordonate system
    rotation = Rot.from_quat([quat[0], quat[1], quat[2], quat[3]])
    return rotation.apply(vect)


def WindVect(quat, SPData):
    """
    Calculate the WindVect from the mooving coordonate sytem to the global coordonate system
    """
    vect = [np.sin(SPData[3]), - np.cos(SPData[3]), np.sin(SPData[2])]  # WindVector in the mooving coordonate system
    vect = (vect / np.linalg.norm(vect)) * (SPData[0] / 10)
    rotation = Rot.from_quat([quat[0], quat[1], quat[2], quat[3]])
    return rotation.apply(vect)


def GetOptiTrackData(lalist, rb_id):
    """
    Get the Data of the optiTrack with a right format and directly from eht streamed source
    """
    for (ac_id, pos, quat, valid) in lalist:
        if ac_id == rb_id:
            return np.array(pos), np.array(quat)
        else:
            continue


def convert_to_beta(v1, v2):
    """
    Calculate the Beta angle to display it in the real time section of the interface
    """
    np.seterr(divide='ignore', invalid='ignore')
    vect1 = [v1[0], v1[1], 0] / np.linalg.norm([v1[0], v1[1], 0])
    vect2 = [v2[0], v2[1], 0] / np.linalg.norm([v2[0], v2[1], 0])
    if v1[0] < v2[0]:
        return np.degrees(np.arccos(np.dot(vect1, vect2)))
    else:
        return -np.degrees(np.arccos(np.dot(vect1, vect2)))


def convert_to_alpha(v1, v2):
    """
    Calculate the Aplha angle to display it in the real time section of the interface
    """
    np.seterr(divide='ignore', invalid='ignore')
    vect1 = [0, v1[1], v1[2]] / np.linalg.norm([0, v1[1], v1[2]])
    vect2 = [0, v2[1], v2[2]] / np.linalg.norm([0, v2[1], v2[2]])
    if v1[2] < v2[2]:
        return +np.degrees(np.arccos(np.dot(vect1, vect2)))
    else:
        return -np.degrees(np.arccos(np.dot(vect1, vect2)))


def pitch_angle(v1):
    """
    Calculate the Pitch angle to display it in the real time section of the interface
    """
    np.seterr(divide='ignore', invalid='ignore')
    vect1 = [v1[0], v1[1], 0] / np.linalg.norm([v1[0], v1[1], 0])
    vect2 = [v1[0], v1[1], v1[2]] / np.linalg.norm([v1[0], v1[1], v1[2]])
    if v1[2] < 0:
        return +np.degrees(np.arccos(np.dot(vect1, vect2)))
    else:
        return -np.degrees(np.arccos(np.dot(vect1, vect2)))


class SerialTutorial:
    """
    Class SerialTutorial that uses pprzlink.serial.SerialMessagesInterface to send
    PING messages on a serial device and monitors incoming messages.
    It respond to PING messages with PONG messages.
    """

    # Construction of the SerialTutorial object
    def __init__(self, dev='COM7', baud=230400):
        self.serial_interface = pprzlink.serial.SerialMessagesInterface(
            callback=self.storeSmartProbeData,  # callback function
            device=dev,  # serial device
            baudrate=baud,  # baudrate
            # interface_id=args.ac_id,  # id of the aircraft
        )
        # self.ac_id = args.ac_id
        self.baudrate = baud
        self.device = dev
        self.smartprobeData = np.array([0, 0, 0, 0])

        # Main loop of the tutorial

    def run(self):
        print("Starting serial interface on %s at %i baud" % (self.device, self.baudrate))

        try:
            self.serial_interface.start()
            # give the thread some time to properly start
            time.sleep(0.1)
        except (KeyboardInterrupt, SystemExit):
            print('Shutting down...')
            self.serial_interface.stop()
            exit()

    # Callback function that process incoming messages
    def process_incoming_message(self, source, pprz_message):
        print("Received message from %i: %s" % (source, pprz_message))

    def storeSmartProbeData(self, source, pprz_message):
        a = str(pprz_message)[20:]
        a = a.split(',')
        TAS = float(a[10].split(' ')[-1])
        EAS = float(a[11].split(' ')[-1])
        alpha = float(a[12].split(' ')[-1])
        beta = float(a[13].split(' ')[-1])
        self.smartprobeData = np.array([TAS, EAS, alpha, beta])


def fonctionvent(X, Y, Z):
    """
    Fonction that is used to simulate the Wind field in the space, and that is suppoosed to be found by the GPR.
    """
    u = -X * (1 - Y)
    v = -(1 - Y)
    w = -Z

    return np.array([u, v, w])


def VarianceToColor(variance):
    """
    Configures the colors of the Variance displayed on the 3D interface.

    Note: a and b have to be configured depending on the "fonction vent"
    they represent the max et min variance of the space
    """
    a = 0
    b = 0.2
    rgb = colorsys.hsv_to_rgb((b - variance) / (3 * (b - a)), 1.0, 1.0)
    return tuple([round(255 * x) for x in rgb])
