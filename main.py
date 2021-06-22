import sys
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph.opengl as gl
from PyQt5 import uic
import numpy as np
from UsefulFunction import tail, SmartProbeVect





class app_1(QtWidgets.QMainWindow):
    def __init__(self):
        super(app_1, self).__init__()
        uic.loadUi('config/interfacefinal.ui', self)
        self.setWindowTitle('Test GL app')
        self.show()

        axis = gl.GLAxisItem()
        self.mainView.addItem(axis)
        ground = gl.GLGridItem()
        ground.scale(1, 1, 0)
        self.mainView.addItem(ground)

        # pos = np.array([tail('Data/OptiTrackData.csv', 1)[0][1:4]])
        # self.sp = gl.GLScatterPlotItem(pos=pos, size=5, pxMode=True)
        # self.mainView.addItem(self.sp)

        Xdot = tuple(tail('Data/OptiTrackData.csv', 1)[0][1:4])
        Ydotmouv = SmartProbeVect()
        Ydot = tuple([Xdot[0] + Ydotmouv[0], Xdot[1] + Ydotmouv[1], Xdot[2] + Ydotmouv[2]])

        SPpos = np.array([Xdot, Ydot])

        self.SmartProbe = gl.GLLinePlotItem(pos=SPpos, width=1, antialias=False)
        self.mainView.addItem(self.SmartProbe)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        # pos = np.array([tail('Data/OptiTrackData.csv', 1)[0][1:4]])
        # self.sp.setData(pos=pos)
        Xdot = tuple(tail('Data/OptiTrackData.csv', 1)[0][1:4])
        Ydotmouv = SmartProbeVect()
        Ydot = tuple([Xdot[0] + Ydotmouv[0], Xdot[1] + Ydotmouv[1], Xdot[2] + Ydotmouv[2]])
        SPpos = np.array([Xdot, Ydot])
        self.SmartProbe.setData(pos=SPpos)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wid = app_1()
    sys.exit(app.exec_())
