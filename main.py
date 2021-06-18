import sys
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph.opengl as gl
from PyQt5 import uic
import numpy as np
import csv


def tail(fn, n):
    with open(fn, 'r') as f:
        f.readline()
        lines = f.readlines()
    return [list(map(float, line.strip().split(','))) for line in lines[-n:]]


def readOptiTrackData(fn):
    with open(fn, 'r') as f:
        reader = csv.reader(f)
        b = next(reader)
        c = b
        return [list(map(eval, c))]


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

        # pos = np.array([tail('optiTrackData.csv', 1)[0][1:4]])
        pos = np.array([readOptiTrackData('Data/OptiTrackData.csv')[0][1:4]])
        self.sp = gl.GLScatterPlotItem(pos=pos, size=5, pxMode=True)
        self.mainView.addItem(self.sp)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        pos = np.array([readOptiTrackData('Data/OptiTrackData.csv')[0][1:4]])
        self.sp.setData(pos=pos)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wid = app_1()
    sys.exit(app.exec_())
