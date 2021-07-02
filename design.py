from time import sleep
from multiprocessing import Process
import pyqtgraph.opengl as gl
import numpy as np
from pyqtgraph import Vector
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5 import uic
from UsefulFunction import tail, SmartProbeVect, WindVect, TrackClass, VisuClass, GPRClass

# def work(foo):
#     foo.work()
#
# pool.apply_async(work,args=(foo,))
Track = TrackClass()
Visu = VisuClass()
GPR = GPRClass(espace=Visu.Points)


class app_1(QtWidgets.QMainWindow):
    def __init__(self):
        super(app_1, self).__init__()
        uic.loadUi('config/interfacefinal.ui', self)
        self.setWindowTitle('Test GL app')
        self.ButtonConnection()
        self.axis()
        self.WallsAndGround()
        self.WindVect()
        self.quadrillageView()
        self.SmartProbeView()
        # self.secondaryView.getPlotItem().hideAxis('bottom')
        # self.secondaryView.getPlotItem().hideAxis('left')
        self.updateTimer()
        self.ComputeThreadCall()

    def ComputeThreadCall(self):
        self.computeThread = QtCore.QThread()
        self.worker = computeThread()
        self.worker.moveToThread(self.computeThread)
        self.computeThread.started.connect(self.worker.run)
        self.computeThread.start()

    def quadrillageView(self):
        quadrillage = gl.GLScatterPlotItem(pos=Visu.Points, size=2)
        self.mainView.addItem(quadrillage)

    def SmartProbeView(self):
        self.SmartProbe = gl.GLLinePlotItem(width=1, antialias=False)
        self.mainView.addItem(self.SmartProbe)

    def TrackView(self):
        self.Trackplot = gl.GLLinePlotItem(width=1, antialias=False)
        self.mainView.addItem(self.Trackplot)

    def FieldWindVectView(self):
        self.WindVectList = []
        for i in range((self.Visu.resolution) ** 3):
            self.WindVectList.append(gl.GLLinePlotItem(width=0.5, antialias=False))
            self.mainView.additem(self.WindVectList[i])

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
        ground = gl.GLGridItem(size=Vector(10, 10, 0), antialias=True)
        Wall1 = gl.GLGridItem(size=Vector(10, 10, 0), antialias=True)
        Wall2 = gl.GLGridItem(size=Vector(10, 10, 0), antialias=True)
        Wall1.rotate(90, 0, 1, 0)
        Wall2.rotate(90, 1, 0, 0)
        Wall1.translate(-5, 0, 5)
        Wall2.translate(0, -5, 5)
        self.mainView.addItem(ground)
        self.mainView.addItem(Wall1)
        self.mainView.addItem(Wall2)

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

    def CalculTraitement(self):
        while True:
            OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
            smartprobeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
            mainPos = OptiData[1:4]
            quat = OptiData[4:8]
            self.Track.TrackStorage(mainPos, WindVect(quat, smartprobeData))
            self.GPR.setDataForGPR(self.Track.wholeTrack, self.Track.wholeWindTrack)

    def aquisitionData(self):
        pass

    def update(self):
        """Il s'agit de la fonction callback de thread principal de Qt"""

        """Il s'agit de la partie d'aqcisition de la Data"""
        OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
        smartprobeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
        mainPos = OptiData[1:4]
        quat = OptiData[4:8]
        ############################################################
        # TODO : Afficher les vecteurs dans le champs !

        # for i in range((self.Visu.resolution)**3):
        #    self.WindVectList[i].setData(mainPos)
        ############################################################
        """Il s'agit de la partie SmartProbe"""
        SPP2prim = SmartProbeVect(quat)
        SPP2 = tuple([mainPos[0] + SPP2prim[0], mainPos[1] + SPP2prim[1], mainPos[2] + SPP2prim[2]])
        SPpos = np.array([mainPos, SPP2])
        self.SmartProbe.setData(pos=SPpos)

        ############################################################
        """Il s'agit de la partie du Vecteur vent avec le bout du vecteur"""
        WindP2prim = WindVect(quat, smartprobeData)
        WindP2 = tuple([mainPos[0] + WindP2prim[0], mainPos[1] + WindP2prim[1], mainPos[2] + WindP2prim[2]])
        Windpos = np.array([mainPos, WindP2])
        self.Wind.setData(pos=Windpos)

        WindDotpos = np.array([WindP2])
        self.WindVectDot.setData(pos=WindDotpos)
        ############################################################
        """Il s'agit de la partie position temps reel"""
        self.SpeedLCDNumber.display(smartprobeData[0])
        self.angleofattackLCDNumber.display(np.degrees(smartprobeData[2]))
        self.pitchangleLCDNumber.display(np.degrees(smartprobeData[2]))
        self.sideslipLCDNumber.display(np.degrees(smartprobeData[3]))
        self.xyzPosition.setText("<html><head/><body><p>x=%0.001f</p><p>y=%0.001f</p><p>z=%0.001f</p></body></html>"
                                 % (mainPos[0], mainPos[1], mainPos[2]))
        ############################################################

        # TrackStorage and ploting
        # Track.TrackStorage(mainPos, WindVect(quat, smartprobeData))
        # self.Trackplot.setData(pos=np.array(Track.recentTrack))


class computeThread(QtCore.QObject):
    def __init__(self):
        super(computeThread, self).__init__()

    def run(self):
        while True:
            OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
            smartprobeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
            mainPos = OptiData[1:4]
            quat = OptiData[4:8]
            Track.TrackStorage(mainPos, WindVect(quat, smartprobeData))
            GPR.setDataForGPR(Track.wholeTrack, Track.wholeWindTrack)
            GPR.predictWindGPR()
            print(Visu.windDistribution(GPR.Yx_pred, GPR.Yy_pred, GPR.Yz_pred))
