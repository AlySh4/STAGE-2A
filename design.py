import pyqtgraph.opengl as gl
import numpy as np
from pyqtgraph import Vector
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5 import uic
from UsefulFunction import tail, SmartProbeVect, WindVect, TrackClass, VisuClass, GPRClass


class app_1(QtWidgets.QMainWindow):
    def __init__(self):
        super(app_1, self).__init__()
        self.timer = QtCore.QTimer()
        self.Track = TrackClass()
        self.GPR = GPRClass()
        self.Visu = VisuClass()
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

    def quadrillageView(self):
        quadrillage = gl.GLScatterPlotItem(pos=self.Visu.Points, size=2)
        self.mainView.addItem(quadrillage)

    def SmartProbeView(self):
        self.SmartProbe = gl.GLLinePlotItem(width=1, antialias=True)
        self.mainView.addItem(self.SmartProbe)

    def TrackView(self):
        self.Trackplot = gl.GLLinePlotItem(width=1, antialias=True)
        self.mainView.addItem(self.Trackplot)

    def WindVect(self):
        self.WindVectDot = gl.GLScatterPlotItem(size=15, color=(1, 0, 0, 1))
        self.mainView.addItem(self.WindVectDot)
        self.Wind = gl.GLLinePlotItem(width=3, color=(1, 0, 0, 1), antialias=True)
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
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
        smartprobeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
        mainPos = OptiData[1:4]
        quat = OptiData[4:8]
        # mainPos = np.array(tail('Data/OptiTrackData.csv', 1)[0][1:4])
        # quat = np.array(tail('Data/OptiTrackData.csv', 1)[0][4:8])
        # SPP2prim = SmartProbeVect(quat)

        # SPP2 = tuple([mainPos[0] + SPP2prim[0], mainPos[1] + SPP2prim[1], mainPos[2] + SPP2prim[2]])
        # SPpos = np.array([mainPos, SPP2])

        WindP2prim = WindVect(quat, smartprobeData)
        WindP2 = tuple([mainPos[0] + WindP2prim[0], mainPos[1] + WindP2prim[1], mainPos[2] + WindP2prim[2]])
        Windpos = np.array([mainPos, WindP2])

        self.SpeedLCDNumber.display(smartprobeData[0])
        self.angleofattackLCDNumber.display(np.degrees(smartprobeData[2]))
        self.pitchangleLCDNumber.display(np.degrees(smartprobeData[2]))
        self.sideslipLCDNumber.display(np.degrees(smartprobeData[3]))

        WindDotpos = np.array([WindP2])

        self.WindVectDot.setData(pos=WindDotpos)
        # self.SmartProbe.setData(pos=SPpos)
        self.Wind.setData(pos=Windpos)

        self.xyzPosition.setText("<html><head/><body><p>x=%0.001f</p><p>y=%0.001f</p><p>z=%0.001f</p></body></html>"
                                 % (mainPos[0], mainPos[1], mainPos[2]))

        # TrackStorage and ploting
        # self.Track.TrackStorage(mainPos)
        # self.Trackplot.setData(pos=np.array(self.Track.list))
