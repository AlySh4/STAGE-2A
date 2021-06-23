import sys
import pyqtgraph.opengl as gl
import numpy as np

from pyqtgraph import Vector
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5 import uic
from UsefulFunction import tail, SmartProbeVect, WindVect


class app_1(QtWidgets.QMainWindow):
    def __init__(self):
        super(app_1, self).__init__()
        uic.loadUi('config/interfacefinal.ui', self)
        self.setWindowTitle('Test GL app')
        self.show()

        # buttonConnection
        self.xOyPButton.clicked.connect(self.xOyView)
        self.xOzPButton.clicked.connect(self.xOzView)
        self.yOzPButton.clicked.connect(self.yOzView)
        self.resetPButton.clicked.connect(self.resetView)

        axis = gl.GLAxisItem()
        self.mainView.addItem(axis)

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

        self.SmartProbe = gl.GLLinePlotItem(width=1, antialias=True)
        self.mainView.addItem(self.SmartProbe)

        self.WindVectDot = gl.GLScatterPlotItem(size=15, color=(1, 0, 0, 1))
        self.mainView.addItem(self.WindVectDot)

        self.Wind = gl.GLLinePlotItem(width=3, color=(1, 0, 0, 1), antialias=True)
        self.mainView.addItem(self.Wind)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        P1 = tuple(tail('Data/OptiTrackData.csv', 1)[0][1:4])
        SPP2prim = SmartProbeVect()
        SPP2 = tuple([P1[0] + SPP2prim[0], P1[1] + SPP2prim[1], P1[2] + SPP2prim[2]])
        SPpos = np.array([P1, SPP2])

        WindP2prim = WindVect()
        WindP2 = tuple([P1[0] + WindP2prim[0], P1[1] + WindP2prim[1], P1[2] + WindP2prim[2]])
        Windpos = np.array([P1, WindP2])

        smartprobeData = tail('Data/SmartProbeData.csv', 1)[0]

        self.SpeedLCDNumber.display(smartprobeData[0])
        self.angleofattackLCDNumber.display(np.degrees(smartprobeData[2]))
        self.pitchangleLCDNumber.display(np.degrees(smartprobeData[2]))
        self.sideslipLCDNumber.display(np.degrees(smartprobeData[3]))

        WindDotpos = np.array([WindP2])

        self.WindVectDot.setData(pos=WindDotpos)
        self.SmartProbe.setData(pos=SPpos)
        self.Wind.setData(pos=Windpos)

    def xOyView(self):
        self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=90, azimuth=270, distance=5)  # xOy

    def xOzView(self):
        self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=0, azimuth=270, distance=5)  # xOz

    def yOzView(self):
        self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=0, azimuth=360, distance=5)  # yOz

    def resetView(self):
        self.mainView.setCameraPosition(pos=Vector(0, 0, 0), elevation=30, azimuth=45, distance=10)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wid = app_1()
    sys.exit(app.exec_())
