from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

import pprzlink.serial
import pprzlink.messages_xml_map as messages_xml_map
import pprzlink.message as message
import time
import csv


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
    def __init__(self, d=5, x1=-2, x2=1, y1=-5, y2=1, z1=0, z2=1):
        self.d = d
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z1
        self.resX = int(abs(x2 - x1) * d)
        self.resY = int(abs(y2 - y1) * d)
        self.resZ = int(abs(z2 - z1) * d)
        self.nbpoint = self.resX * self.resY * self.resZ
        self.X = np.linspace(x1, x2, self.resX)
        self.Y = np.linspace(y1, y2, self.resY)
        self.Z = np.linspace(z1, z2, self.resZ)
        self.Points = self.ListePoints()
        self.Xcut = np.insert(np.transpose([np.tile(self.Y, self.resZ), np.repeat(self.Z, self.resY)]), 0,
                              [0] * (self.resY * self.resZ), axis=1)
        self.Ycut = np.insert(np.transpose([np.tile(self.X, self.resZ), np.repeat(self.Z, self.resX)]), 1,
                              [0] * (self.resX * self.resZ), axis=1)
        self.Zcut = np.insert(np.transpose([np.tile(self.X, self.resY), np.repeat(self.Y, self.resX)]), 2,
                              [0] * (self.resX * self.resY), axis=1)

    def ListePoints(self):
        P = []
        for x in self.X:
            for y in self.Y:
                for z in self.Z:
                    P.append((x, y, z))
        return np.array(P)

    def setCut(self, componante, cut):
        if componante == 'x':
            self.Xcut[:, 0] = [cut] * (self.resY * self.resZ)
        if componante == 'y':
            self.Ycut[:, 1] = [cut] * (self.resX * self.resZ)
        if componante == 'z':
            self.Ycut[:, 2] = [cut] * (self.resX * self.resY)

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
        self.espace = espace

    def setDataForGPR(self, pos, wind):  # fonction qui permet de récuperer les data pour faire le training

        # if len(pos) > 500 and len(wind) > 500:
        #     self.Xm = pos[-500:]
        #     self.Yx = wind[:, 0][-500:]
        #     self.Yy = wind[:, 1][-500:]
        #     self.Yz = wind[:, 2][-500:]
        # else:
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

    def predictWindSecViewGPR(self, Xcut, Ycut, Zcut):
        self.Xcut_pred = self.gp1.predict(Xcut)
        self.Ycut_pred = self.gp2.predict(Ycut)
        self.Zcut_pred = self.gp3.predict(Zcut)

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


def GetOptiTrackData(list, rb_id):
    for (ac_id, pos, quat, valid) in list:
        if not valid:
            # skip if rigid body is not valid
            continue
        if ac_id == rb_id:
            return np.array(pos), np.array(quat)
        else:
            continue


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

            # while self.serial_interface.isAlive():
            #     self.serial_interface.join(1)

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
        # with open('Data/SmartProbeData.csv', 'w', newline='') as f:
        #     dataWriter = csv.writer(f)
        #     dataWriter.writerow([TAS, EAS, alpha, beta])
        #     f.flush()

# class SecondaryView():
#     def __init__(self):
#     self.graphX=
#     self.graphY=
#     self.graphZ=
