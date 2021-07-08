import sys
import time
from collections import deque
from time import sleep

import pyqtgraph.opengl as gl
import numpy as np
from pyqtgraph import Vector
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5 import uic
from UsefulFunction import tail, SmartProbeVect, WindVect, TrackClass, VisuClass, GPRClass

Track = TrackClass()
Visu = VisuClass()
GPR = GPRClass(espace=Visu.Points)
OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
mainPos = OptiData[1:4]
quaternion = OptiData[4:8]


class app_1(QtWidgets.QMainWindow):
    def __init__(self):
        super(app_1, self).__init__()
        uic.loadUi('config/interfacefinal.ui', self)
        self.setWindowTitle('Test GL app')
        self.ButtonConnection()
        self.axis()
        self.WallsAndGround()
        self.WindVect()
        self.WindDistributionView()
        # self.quadrillageView()
        self.SmartProbeView()
        # self.secondaryView.getPlotItem().hideAxis('bottom')
        # self.secondaryView.getPlotItem().hideAxis('left')
        self.updateTimer()
        # self.updateTimer1()
        self.ComputeThreadCall()
        self.AcquisitionThreadCall()
        self.WindDistributionThreadCall()

    def WindDistributionThreadCall(self):
        self.WindDistributionThread = QtCore.QThread()
        self.worker3 = computeWindDistributionThread()
        self.worker3.moveToThread(self.WindDistributionThread)
        self.WindDistributionThread.started.connect(self.worker3.run)
        self.WindDistributionThread.start()

    def ComputeThreadCall(self):
        self.computeThread = QtCore.QThread()
        self.worker1 = computeThread()
        self.worker1.moveToThread(self.computeThread)
        self.computeThread.started.connect(self.worker1.run)
        self.computeThread.start()

    def AcquisitionThreadCall(self):
        self.aquisitionThread = QtCore.QThread()
        self.worker2 = AcquisitionThread()
        self.worker2.moveToThread(self.aquisitionThread)
        self.aquisitionThread.started.connect(self.worker2.run)
        self.aquisitionThread.start()

    # def quadrillageView(self):
    #     quadrillage = gl.GLScatterPlotItem(pos=Visu.Points, size=2)
    #     self.mainView.addItem(quadrillage)

    def SmartProbeView(self):
        self.SmartProbe = gl.GLLinePlotItem(width=1, antialias=False)
        self.mainView.addItem(self.SmartProbe)

    def TrackView(self):
        self.Trackplot = gl.GLLinePlotItem(width=1, antialias=False)
        self.mainView.addItem(self.Trackplot)

    def WindDistributionView(self):
        # self.WindDistributionList = deque()
        # for i in range(Visu.resolution ** 3):
        #     self.WindDistributionList.append(gl.GLLinePlotItem(width=0.5, color=(0, 255, 0, 1), antialias=False))
        #     self.mainView.addItem(self.WindDistributionList[i])

        # for i in range(Visu.resolution ** 3):
        #     exec("self.WindDistribution{} = gl.GLLinePlotItem(width=0.5, color=(0, 255, 0, 1), antialias=False)".format(
        #         i))
        #     exec("self.mainView.addItem(self.WindDistribution{})".format(i))

        # self.WindDistribution = gl.GLScatterPlotItem(color=(0, 255, 0, 1), size = 5)
        # self.mainView.addItem(self.WindDistribution)

        self.WindDistribution = gl.GLLinePlotItem(color=(0, 255, 0, 1), width=0.5, mode='lines')
        self.mainView.addItem(self.WindDistribution)

    def WindVect(self):
        self.WindVectDot = gl.GLScatterPlotItem(size=15, color=(1, 0, 0, 1))
        self.mainView.addItem(self.WindVectDot)
        self.Wind = gl.GLLinePlotItem(width=3, color=(1, 0, 0, 1), antialias=False)
        self.mainView.addItem(self.Wind)

    def xOyView(self):
        self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=90, azimuth=270, distance=5)  # xOy

    def xOzView(self):
        self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=0, azimuth=270, distance=5)  # xOz

    def yOzView(self):
        self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=0, azimuth=360, distance=5)  # yOz

    def resetView(self):
        self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=30, azimuth=45, distance=10)

    def WallsAndGround(self):
        ground = gl.GLGridItem(size=Vector(10, 10, 0), antialias=False)
        # Wall1 = gl.GLGridItem(size=Vector(10, 10, 0), antialias=True)
        # Wall2 = gl.GLGridItem(size=Vector(10, 10, 0), antialias=True)
        # Wall1.rotate(90, 0, 1, 0)
        # Wall2.rotate(90, 1, 0, 0)
        # Wall1.translate(-5, 0, 5)
        # Wall2.translate(0, -5, 5)
        self.mainView.addItem(ground)
        # self.mainView.addItem(Wall1)
        # self.mainView.addItem(Wall2)

    def ButtonConnection(self):
        self.xOyPButton.clicked.connect(self.xOyView)
        self.xOzPButton.clicked.connect(self.xOzView)
        self.yOzPButton.clicked.connect(self.yOzView)
        self.resetPButton.clicked.connect(self.resetView)

    def axis(self):
        self.mainView.addItem(gl.GLAxisItem())

    def updateTimer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start()

    # def updateTimer1(self):
    #     self.timer1 = QtCore.QTimer()
    #     self.timer1.timeout.connect(self.update1)
    #     self.timer1.start(3000)
    #
    # def update1(self):
    #     for i in range(Visu.resolution ** 3):
    #         self.WindDistributionList[i].setData(
    #             pos=np.array([Visu.Points[i], Visu.windDistribution(GPR.Yx_pred, GPR.Yy_pred, GPR.Yz_pred)[i]]))

    def update(self):
        ############################################################
        """Il s'agit de la partie SmartProbe"""
        SPP2prim = SmartProbeVect(quaternion)
        SPP2 = tuple([mainPos[0] + SPP2prim[0], mainPos[1] + SPP2prim[1], mainPos[2] + SPP2prim[2]])
        SPpos = np.array([mainPos, SPP2])
        self.SmartProbe.setData(pos=SPpos)

        ############################################################
        """Il s'agit de la partie du Vecteur vent avec le bout du vecteur"""
        WindP2prim = WindVect(quaternion, smartProbeData)
        WindP2 = tuple([mainPos[0] + WindP2prim[0], mainPos[1] + WindP2prim[1], mainPos[2] + WindP2prim[2]])
        Windpos = np.array([mainPos, WindP2])
        self.Wind.setData(pos=Windpos)

        WindDotpos = np.array([WindP2])
        self.WindVectDot.setData(pos=WindDotpos)
        ############################################################
        """Il s'agit de la partie position temps reel"""
        self.SpeedLCDNumber.display(smartProbeData[0])
        self.angleofattackLCDNumber.display(np.degrees(smartProbeData[2]))
        self.pitchangleLCDNumber.display(np.degrees(smartProbeData[2]))
        self.sideslipLCDNumber.display(np.degrees(smartProbeData[3]))
        self.xyzPosition.setText("<html><head/><body><p>x=%0.001f</p><p>y=%0.001f</p><p>z=%0.001f</p></body></html>"
                                 % (mainPos[0], mainPos[1], mainPos[2]))
        ############################################################

        # TrackStorage and ploting
        # Track.TrackStorage(mainPos, WindVect(quat, smartprobeData))
        # self.Trackplot.setData(pos=np.array(Track.recentTrack))


class AcquisitionThread(QtCore.QObject):
    def __init__(self):
        super(AcquisitionThread, self).__init__()

    def run(self):
        global OptiData, smartProbeData, mainPos, quaternion
        while True:
            try:
                OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
                smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
                mainPos = OptiData[1:4]
                quaternion = OptiData[4:8]
                Track.TrackStorage(mainPos, WindVect(quaternion, smartProbeData))
            except IndexError:
                continue


class computeThread(QtCore.QObject):
    def __init__(self):
        super(computeThread, self).__init__()

    def run(self):
        while True:
            try:
                GPR.setDataForGPR(Track.wholeTrack, Track.wholeWindTrack)
                GPR.predictWindGPR()
            # print(Visu.windDistribution(GPR.Yx_pred, GPR.Yy_pred, GPR.Yz_pred))
            except ValueError:
                continue


class computeWindDistributionThread(QtCore.QObject):
    def __init__(self):
        super(computeWindDistributionThread, self).__init__()

    # def run(self):
    #     while True:
    #         start = time.time()
    #         try:
    #             for i in range(Visu.resolution ** 3):
    #                 wid.WindDistributionList[i].setData(
    #                     pos=np.array([Visu.Points[i], Visu.windDistribution(GPR.Yx_pred, GPR.Yy_pred, GPR.Yz_pred)[i]]))
    #         except TypeError:
    #             continue
    #         end = time.time()
    #         print(end-start)

    # def run(self):
    #     while True:
    #         try:
    #             for i in range(Visu.resolution ** 3):
    #                 exec(
    #                     "wid.WindDistribution{}.setData(pos=np.array([Visu.Points[i], Visu.windDistribution(GPR.Yx_pred, GPR.Yy_pred, GPR.Yz_pred)[i]]))".format(
    #                         i))
    #         except TypeError:
    #             continue

    # def run(self):
    #     while True:
    #         try:
    #             bj = np.empty((0,3))
    #             for i in range(Visu.resolution ** 3):
    #                 a = Visu.Points[i]
    #                 b = Visu.windDistribution(GPR.Yx_pred/10, GPR.Yy_pred/10, GPR.Yz_pred/10)[i]
    #                 bj = np.concatenate((bj, a.reshape((1,3)), (np.array([(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2])).reshape((1,3)), b.reshape((1,3))))
    #             wid.WindDistribution.setData(pos=bj)
    #         except TypeError:
    #             continue

    def run(self):
        while True:
            try:
                pos = np.empty((0, 3))
                for i in range(Visu.resolution ** 3):
                    a = Visu.Points[i]
                    b = Visu.windDistribution(GPR.Yx_pred / 10, GPR.Yy_pred / 10, GPR.Yz_pred / 10)[i]
                    pos = np.concatenate((pos, a.reshape((1, 3)), b.reshape(1, 3)))
                wid.WindDistribution.setData(pos=pos)
            except TypeError:
                continue


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wid = app_1()
    wid.show()
    sys.exit(app.exec_())
