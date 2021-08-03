import sys
from time import sleep
import pyqtgraph.opengl as gl
import numpy as np
from pyqtgraph import Vector, ImageItem, HistogramLUTItem
# from pyqtgraph.graphicsItems import ColorBarItem
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from OriginalTemplates.NatNetClient import NatNetClient
from UsefulFunction import tail, SmartProbeVect, WindVect, TrackClass, VisuClass, GPRClass, GetOptiTrackData, \
    SerialTutorial, convert_to_alpha, convert_to_beta, pitch_angle

dev = 0  # 0, 1 ou 2

Track = TrackClass()
Visu = VisuClass()
GPR = GPRClass(espace=Visu.Points)


class app_1(QtWidgets.QMainWindow):
    def __init__(self, SPid=224, dp=1511, cp=1510, server="192.168.1.235"):

        if dev == 0:
            pass

        if dev == 1:
            self.natnet = NatNetClient(
                server=server,
                dataPort=dp,
                commandPort=cp)
            self.natnet.run()

        if dev == 2:
            self.serial = SerialTutorial()
            self.serial.run()
            self.natnet = NatNetClient(
                server=server,
                dataPort=dp,
                commandPort=cp)
            self.natnet.run()

        super(app_1, self).__init__()
        self.SPid = SPid
        uic.loadUi('config/interfacefinal.ui', self)
        self.setWindowTitle('Test GL app')
        self.ButtonConnection()
        self.ButtonStatusInit()
        self.axis()
        self.WallsAndGround()
        self.WindVect()
        self.WindDistributionView()
        self.SmartProbeView()

        self.updateTimer()
        # self.TrackView()
        self.secondaryViewplot()

    def ButtonStatusInit(self):
        self.stateGPRbutton = True
        self.GPRstatut.setStyleSheet("background-color: green")
        self.GPRstatut.pressed.connect(self.ComputeThreadCall)
        self.counter = False

        self.stateGPRShowing = True
        self.GPRShowingButton.setStyleSheet("background-color: gray")
        self.GPRShowingButton.pressed.connect(self.WindDistributionShowingstate)

    def WindDistributionShowingstate(self):
        if self.counter:
            if self.stateGPRShowing:
                self.WindDistribution.setData(color=(0, 0, 0, 0))
                self.GPRShowingButton.setStyleSheet("background-color: red")
                self.stateGPRShowing = False
            else:
                self.GPRShowingButton.setStyleSheet("background-color: green")
                self.WindDistribution.setData(color=(0, 255, 127, 1))
                self.stateGPRShowing = True

    def ComputeThreadCall(self):
        self.counter = True
        if self.stateGPRbutton:
            self.computeThread = QtCore.QThread()
            self.worker1 = computeThread()
            self.worker1.moveToThread(self.computeThread)
            self.computeThread.started.connect(self.worker1.run)
            self.computeThread.start()
            self.stateGPRbutton = False
            self.GPRstatut.setStyleSheet("background-color: red")
            self.GPRstatut.setText("Stop GPR")
            self.GPRShowingButton.setStyleSheet("background-color: green")
        else:
            self.stateGPRbutton = True
            self.worker1.stop()
            self.computeThread.quit()
            self.computeThread.wait()
            self.GPRstatut.setStyleSheet("background-color: green")
            self.GPRstatut.setText("Start GRP")

    def secondaryViewplot(self):
        self.Xview = ImageItem()
        self.Yview = ImageItem()
        self.Zview = ImageItem()

        bar = HistogramLUTItem(image=self.Xview)
        self.Xplotview.addItem(bar)

        self.sliderXcut.setRange(0, Visu.resX - 1)
        self.sliderYcut.setRange(0, Visu.resY - 1)
        self.sliderZcut.setRange(0, Visu.resZ - 1)

        def changeValueX(value):
            if self.counter:
                # realValue = value / 10
                # self.labelXcut.setText("<html><head/><body><p>x = %0.1f</p></body></html>" % realValue)
                # Visu.setCut('x', realValue)

                self.labelXcut.setText("<html><head/><body><p>x = %0.1f</p></body></html>" % Visu.X[value])
                Visu.newsetCut('x', value)
                GPR.WindSecViewGPR('x', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuut, Visu.ycuut, Visu.zcuut)
                wid.Xview.setImage(image=GPR.Xcut_pred.reshape((Visu.resY, Visu.resZ)), levels=(0, 2))

        def changeValueY(value):
            if self.counter:
                # realValue = value / 10
                # self.labelYcut.setText("<html><head/><body><p>y = %0.1f</p></body></html>" % realValue)
                # Visu.setCut('y', realValue)

                self.labelYcut.setText("<html><head/><body><p>y = %0.1f</p></body></html>" % Visu.Y[value])
                Visu.newsetCut('y', value)
                GPR.WindSecViewGPR('y', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuut, Visu.ycuut, Visu.zcuut)
                wid.Yview.setImage(image=GPR.Ycut_pred.reshape((Visu.resZ, Visu.resX)), levels=(0, 2))

        def changeValueZ(value):
            if self.counter:
                # realValue = value / 10
                # self.labelZcut.setText("<html><head/><body><p>z = %0.1f</p></body></html>" % realValue)
                # Visu.setCut('z', realValue)

                self.labelZcut.setText("<html><head/><body><p>z = %0.1f</p></body></html>" % Visu.Z[value])
                Visu.newsetCut('z', value)
                GPR.WindSecViewGPR('z', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuut, Visu.ycuut, Visu.zcuut)
                wid.Zview.setImage(image=GPR.Zcut_pred.reshape((Visu.resX, Visu.resY)), levels=(0, 2))

        self.sliderZcut.valueChanged.connect(changeValueZ)
        self.sliderXcut.valueChanged.connect(changeValueX)
        self.sliderYcut.valueChanged.connect(changeValueY)

        # self.Xplotview.getPlotItem().hideAxis('bottom')
        # self.Xplotview.getPlotItem().hideAxis('left')
        # self.Yplotview.getPlotItem().hideAxis('bottom')
        # self.Yplotview.getPlotItem().hideAxis('left')
        # self.Zplotview.getPlotItem().hideAxis('bottom')
        # self.Zplotview.getPlotItem().hideAxis('left')

        self.Xplotview.addItem(self.Xview)
        self.Yplotview.addItem(self.Yview)
        self.Zplotview.addItem(self.Zview)

    def SmartProbeView(self):
        self.SmartProbe = gl.GLLinePlotItem(width=1, antialias=False)
        self.mainView.addItem(self.SmartProbe)

    def TrackView(self):
        self.Trackplot = gl.GLLinePlotItem(width=1, antialias=False)
        self.mainView.addItem(self.Trackplot)

    def WindDistributionView(self):
        self.WindDistribution = gl.GLLinePlotItem(color=(0, 255, 127, 1), width=0.1, mode='lines')
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

        self.mainView.addItem(gl.GLAxisItem(size=Vector(1, 1, 1)))
        self.XcordLabel = gl.GLLinePlotItem(color=(231, 37, 131, 0.4), width=0.3, mode='lines', antialias=True)
        self.YcordLabel = gl.GLLinePlotItem(color=(231, 37, 131, 0.4), width=0.3, mode='lines', antialias=True)
        self.ZcordLabel = gl.GLLinePlotItem(color=(231, 37, 131, 0.4), width=0.3, mode='lines', antialias=True)
        self.XcordLabel.setData(pos=np.array([[0.9, 0.05, 0], [1, 0.15, 0], [0.9, 0.15, 0], [1, 0.05, 0]]))
        self.YcordLabel.setData(pos=np.array([[-0.05, 0.9, 0], [-0.15, 1, 0], [-0.15, 0.9, 0], [-0.1, 0.95, 0]]))
        self.ZcordLabel.setData(pos=np.array([[0, 0.05, 1], [0, 0.15, 1], [0, 0.15, 1], [0, 0.05, 0.9], [0, 0.05, 0.9],
                                              [0, 0.15, 0.9]]))
        self.mainView.addItem(self.XcordLabel)
        self.mainView.addItem(self.YcordLabel)
        self.mainView.addItem(self.ZcordLabel)

    def updateTimer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        try:
            if dev == 0:
                OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
                smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
                mainPos = OptiData[1:4]
                quaternion = OptiData[4:8]

            if dev == 1:
                OptiData = GetOptiTrackData(self.natnet.rigidBodyList, wid.SPid)
                smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
                mainPos = OptiData[0]
                quaternion = OptiData[1]

            if dev == 2:
                OptiData = GetOptiTrackData(self.natnet.rigidBodyList, wid.SPid)
                smartProbeData = self.serial.smartprobeData
                mainPos = OptiData[0]
                quaternion = OptiData[1]

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
            self.xyzPosition.setText("<html><head/><body><p>x=%0.001f</p><p>y=%0.001f</p><p>z=%0.001f</p></body></html>"
                                     % (mainPos[0], mainPos[1], mainPos[2]))
            self.SpeedLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[0]))
            # self.angleofattackLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[2]))
            self.pitchangleLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[2]))
            # self.sideslipeLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[3]))
            self.SpeedLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[0]))
            self.angleofattackLabelValue.setText(
                "<html><head/><body>%0.001f</body></html>" % (convert_to_alpha(SPP2prim, WindP2prim)))
            self.pitchangleLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (pitch_angle(SPP2prim)))
            self.sideslipeLabelValue.setText(
                "<html><head/><body>%0.001f</body></html>" % (convert_to_beta(SPP2prim, WindP2prim)))

            ############################################################

            # TrackStorage and ploting
            # self.Trackplot.setData(pos=np.array(Track.recentTrack))

        except (TypeError, IndexError):
            pass


class computeThread(QtCore.QObject):
    def __init__(self):
        super(computeThread, self).__init__()
        self.isRunning = True

    def run(self):
        while self.isRunning:
            try:

                if dev == 0:
                    OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
                    smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
                    mainPos = OptiData[1:4]
                    quaternion = OptiData[4:8]

                if dev == 1:
                    OptiData = GetOptiTrackData(self.natnet.rigidBodyList, wid.SPid)
                    smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
                    mainPos = OptiData[0]
                    quaternion = OptiData[1]

                if dev == 2:
                    OptiData = GetOptiTrackData(self.natnet.rigidBodyList, wid.SPid)
                    smartProbeData = self.serial.smartprobeData
                    mainPos = OptiData[0]
                    quaternion = OptiData[1]

                Track.TrackStorage(mainPos, WindVect(quaternion, smartProbeData))
                GPR.setDataForGPR(Track.wholeTrack, Track.wholeWindTrack)
                GPR.predictWindGPR()
                pos = np.empty((0, 3))
                a = Visu.Points
                b = Visu.windDistribution(GPR.Yx_pred / 2, GPR.Yy_pred / 2, GPR.Yz_pred / 2)
                for i in range(Visu.nbpoint):
                    pos = np.concatenate((pos, a[i].reshape((1, 3)), b[i].reshape(1, 3)))
                wid.WindDistribution.setData(pos=pos)
                # GPR.predictWindSecViewGPR(Visu.Xcut, Visu.Ycut, Visu.Zcut)
                GPR.WindSecViewGPR('all', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuut, Visu.ycuut, Visu.zcuut)
                wid.Xview.setImage(image=GPR.Xcut_pred.reshape((Visu.resY, Visu.resZ)), levels=(0, 2))
                wid.Yview.setImage(image=GPR.Ycut_pred.reshape((Visu.resZ, Visu.resX)), levels=(0, 2))
                wid.Zview.setImage(image=GPR.Zcut_pred.reshape((Visu.resX, Visu.resY)), levels=(0, 2))
                sleep(1)
            except (NameError, TypeError):
                continue

    def stop(self):
        self.isRunning = False


if __name__ == '__main__':  # TODO: parser pour parametrer le natnet.
    if dev == 0:
        app = QtWidgets.QApplication(sys.argv)
        wid = app_1()
        wid.show()
        sys.exit(app.exec_())

    if dev == 1:
        try:
            app = QtWidgets.QApplication(sys.argv)
            wid = app_1()
            wid.show()
            sys.exit(app.exec_())
        except (KeyboardInterrupt, SystemExit):
            print("Shutting down natnet interfaces...")
            wid.natnet.stop()
        except OSError:
            print("Natnet connection error")
            wid.natnet.stop()
            exit(-1)
    if dev == 2:
        try:
            app = QtWidgets.QApplication(sys.argv)
            wid = app_1()
            wid.show()
            sys.exit(app.exec_())

        except (KeyboardInterrupt, SystemExit):
            print("Shutting down natnet interfaces...")
            wid.serial.serial_interface.stop()
            wid.natnet.stop()
        except OSError:
            print("Natnet connection error")
            wid.serial.serial_interface.stop()
            wid.natnet.stop()
            exit(-1)
