import sys
from time import sleep
import pyqtgraph.opengl as gl
import numpy as np
from pyqtgraph import Vector, ImageItem, colormap
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from OriginalTemplates.NatNetClient import NatNetClient
from UsefulFunction import tail, SmartProbeVect, WindVect, TrackClass, VisuClass, GPRClass, GetOptiTrackData, \
    SerialTutorial

mutex = QtCore.QMutex

Track = TrackClass()
Visu = VisuClass()
GPR = GPRClass(espace=Visu.Points)


class app_1(QtWidgets.QMainWindow):
    def __init__(self, SPid=217, dp=1511, cp=1510, server="192.168.1.231"):
        super(app_1, self).__init__()
        self.SPid = SPid

        ### LA vrai
        # self.serial = SerialTutorial()
        # self.serial.run()
        # self.natnet = NatNetClient(
        #     server=server,
        #     dataPort=dp,
        #     commandPort=cp)
        # self.natnet.run()
        ###

        uic.loadUi('config/interfacefinal.ui', self)
        self.setWindowTitle('Test GL app')
        self.ButtonConnection()
        self.axis()
        self.WallsAndGround()
        self.WindVect()
        self.WindDistributionView()
        self.SmartProbeView()
        # self.Xplotview.getPlotItem().hideAxis('bottom')
        # self.secondaryView.getPlotItem().hideAxis('left')
        self.updateTimer()
        self.ComputeThreadCall()
        # self.testThreadCall()
        # self.TrackView()
        self.secondaryViewplot()

    def secondaryViewplot(self):
        self.Xview = ImageItem(levels=(0, 1))
        self.Yview = ImageItem(levels=(0, 1))
        self.Zview = ImageItem(levels=(0, 1))

        self.slideryOz.setRange(Visu.x1, Visu.x2)
        self.sliderxOz.setRange(Visu.y1, Visu.y2)
        self.sliderxOy.setRange(Visu.z1, Visu.z2)

        self.Xplotview.addItem(self.Xview)
        self.Yplotview.addItem(self.Yview)
        self.Zplotview.addItem(self.Zview)

    def ComputeThreadCall(self):
        self.computeThread = QtCore.QThread()
        self.worker1 = computeThread()
        self.worker1.moveToThread(self.computeThread)
        self.computeThread.started.connect(self.worker1.run)
        self.computeThread.start()

    def SmartProbeView(self):
        self.SmartProbe = gl.GLLinePlotItem(width=1, antialias=False)
        self.mainView.addItem(self.SmartProbe)

    def TrackView(self):
        self.Trackplot = gl.GLLinePlotItem(width=1, antialias=False)
        self.mainView.addItem(self.Trackplot)

    def WindDistributionView(self):
        self.WindDistribution = gl.GLLinePlotItem(color=(0, 255, 0, 1), width=0.1, mode='lines')
        self.mainView.addItem(self.WindDistribution)

    def WindVect(self):
        self.WindVectDot = gl.GLScatterPlotItem(size=15, color=(1, 0, 0, 1))
        self.mainView.addItem(self.WindVectDot)
        self.Wind = gl.GLLinePlotItem(width=3, color=(1, 0, 0, 1), antialias=False)
        self.mainView.addItem(self.Wind)



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
        def xOyView():
            self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=90, azimuth=270, distance=5)  # xOy

        def xOzView():
            self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=0, azimuth=270, distance=5)  # xOz

        def yOzView():
            self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=0, azimuth=360, distance=5)  # yOz

        def resetView():
            self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=30, azimuth=45, distance=10)

        self.xOyPButton.clicked.connect(xOyView)
        self.xOzPButton.clicked.connect(xOzView)
        self.yOzPButton.clicked.connect(yOzView)
        self.resetPButton.clicked.connect(resetView)

    def axis(self):
        self.mainView.addItem(gl.GLAxisItem())

    def updateTimer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        try:
            # les vrais data##############################################
            # OptiData = GetOptiTrackData(self.natnet.rigidBodyList, wid.SPid)
            # smartProbeData = self.serial.smartprobeData
            # mainPos = OptiData[0]
            # quaternion = OptiData[1]

            # pour tester
            OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
            smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
            mainPos = OptiData[1:4]
            quaternion = OptiData[4:8]

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
            # self.Trackplot.setData(pos=np.array(Track.recentTrack))

        except TypeError:
            pass


class computeThread(QtCore.QObject):
    def __init__(self):
        super(computeThread, self).__init__()

    def run(self):
        while True:
            # les vrais data##############################################
            # OptiData = GetOptiTrackData(self.natnet.rigidBodyList, wid.SPid)
            # smartProbeData = self.serial.smartprobeData
            # mainPos = OptiData[0]
            # quaternion = OptiData[1]

            # pour tester
            OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
            smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
            mainPos = OptiData[1:4]
            quaternion = OptiData[4:8]

            Track.TrackStorage(mainPos, WindVect(quaternion, smartProbeData))
            GPR.setDataForGPR(Track.wholeTrack, Track.wholeWindTrack)
            GPR.predictWindGPR()
            pos = np.empty((0, 3))
            a = Visu.Points
            b = Visu.windDistribution(GPR.Yx_pred / 2, GPR.Yy_pred / 2, GPR.Yz_pred / 2)
            for i in range(Visu.nbpoint):
                pos = np.concatenate((pos, a[i].reshape((1, 3)), b[i].reshape(1, 3)))
            wid.WindDistribution.setData(pos=pos)
            GPR.predictWindSecViewGPR(Visu.Xcut, Visu.Ycut, Visu.Zcut)
            wid.Xview.setImage(image=GPR.Xcut_pred.reshape((Visu.resY, Visu.resZ)))
            wid.Yview.setImage(image=GPR.Ycut_pred.reshape((Visu.resX, Visu.resZ)))
            wid.Zview.setImage(image=GPR.Zcut_pred.reshape((Visu.resX, Visu.resY)))


if __name__ == '__main__':  # TODO: parser pour parametrer le natnet.
    # try:
    app = QtWidgets.QApplication(sys.argv)
    wid = app_1()
    wid.show()
    sys.exit(app.exec_())
# except (KeyboardInterrupt, SystemExit):
#     print("Shutting down natnet interfaces...")
#     wid.serial.serial_interface.stop()
#     wid.natnet.stop()
# except OSError:
#     print("Natnet connection error")
#     wid.serial.serial_interface.stop()
#     wid.natnet.stop()
#     exit(-1)
