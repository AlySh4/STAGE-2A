import sys
from time import sleep
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QMessageBox, QFormLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pyqtgraph import Vector, ImageItem, HistogramLUTItem, glColor
from pyqtgraph.graphicsItems import ColorBarItem
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from OriginalTemplates.NatNetClient import NatNetClient
from UsefulFunction import tail, SmartProbeVect, WindVect, TrackClass, VisuClass, GPRClass, GetOptiTrackData, \
    SerialTutorial, convert_to_alpha, convert_to_beta, pitch_angle, fonctionvent, VarianceToColor

dev = 0  # 0, 1 ou 2
Track = TrackClass()
Visu = VisuClass()
GPR = GPRClass(espace=Visu.Points)

uifile_2 = 'config/configurationDialog.ui'
form_2, base_2 = uic.loadUiType(uifile_2)


class ConfigurationWindow(form_2, base_2):
    def __init__(self):
        super(base_2, self).__init__()
        self.setupUi(self)
        self.spinBoxX1.setValue(Visu.x1)
        self.spinBoxX2.setValue(Visu.x2)
        self.spinBoxY1.setValue(Visu.y1)
        self.spinBoxY2.setValue(Visu.y2)
        self.spinBoxZ1.setValue(Visu.z1)
        self.spinBoxZ2.setValue(Visu.z2)
        self.spinBoxRes.setValue(Visu.d)
        self.buttonBox.accepted.connect(self.change)

    def change(self):
        if (self.spinBoxX1.value() < self.spinBoxX2.value()) and (self.spinBoxY1.value() < self.spinBoxY2.value()) and (
                self.spinBoxZ1.value() < self.spinBoxZ2.value()) and self.spinBoxRes.value() > 0:
            Visu.x1 = self.spinBoxX1.value()
            Visu.x2 = self.spinBoxX2.value()
            Visu.y1 = self.spinBoxY1.value()
            Visu.y2 = self.spinBoxY2.value()
            Visu.z1 = self.spinBoxZ1.value()
            Visu.z2 = self.spinBoxZ2.value()
            Visu.d = self.spinBobRes.value()
            self.close()
        else:
            self.info.setText("Please verify your values")


class app_1(QtWidgets.QMainWindow):
    def __init__(self, SPid=224, dp=1511, cp=1510, server="192.168.1.235"):
        self.c = 0
        self.f = 0
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
        uic.loadUi('config/interfacefinal1.ui', self)
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
        # self.secondaryViewplot()
        self.secondaryViewplot()
        self.secondaryViewplotVariance()
        self.tqt()

    def tqt(self):
        self.confwid = ConfigurationWindow()

        def openconf():
            self.confwid.show()

        self.resetButton.clicked.connect(openconf)

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

    def secondaryViewplotVariance(self):
        self.XviewVar = MatplotlibWidget()
        self.YviewVar = MatplotlibWidget()
        self.ZviewVar = MatplotlibWidget()

        self.XimgVar = self.XviewVar.axis.imshow(np.zeros((Visu.resY, Visu.resZ)), cmap=plt.get_cmap('RdYlGn_r'),
                                                 vmin=0,
                                                 vmax=1, interpolation='bilinear')
        self.YimgVar = self.YviewVar.axis.imshow(np.zeros((Visu.resZ, Visu.resX)), cmap=plt.get_cmap('RdYlGn_r'),
                                                 vmin=0,
                                                 vmax=1, interpolation='bilinear')
        self.ZimgVar = self.ZviewVar.axis.imshow(np.zeros((Visu.resX, Visu.resY)), cmap=plt.get_cmap('RdYlGn_r'),
                                                 vmin=0,
                                                 vmax=1, interpolation='bilinear')

        self.XimgVarColorbar = self.XviewVar.figure.colorbar(self.XimgVar)
        self.YimgVarColorbar = self.YviewVar.figure.colorbar(self.YimgVar)
        self.ZimgVarColorbar = self.ZviewVar.figure.colorbar(self.ZimgVar)

        self.layoutXviewVar = QVBoxLayout(self.XplotviewVariance)
        self.layoutYviewVar = QVBoxLayout(self.YplotviewVariance)
        self.layoutZviewVar = QVBoxLayout(self.ZplotviewVariance)

        self.layoutXviewVar.addWidget(self.XviewVar)
        self.layoutYviewVar.addWidget(self.YviewVar)
        self.layoutZviewVar.addWidget(self.ZviewVar)

        self.sliderXcutVariance.setRange(0, Visu.resX - 1)
        self.sliderYcutVariance.setRange(0, Visu.resY - 1)
        self.sliderZcutVariance.setRange(0, Visu.resZ - 1)

        def changeValueXVar(value):
            if self.counter:
                self.labelXcutVariance.setText("<html><head/><body><p>x = %0.1f</p></body></html>" % Visu.X[value])
                Visu.newsetCutVariance('x', value)
                GPR.VarianceSecViewGPR('z', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuutVariance, Visu.ycuutVariance,
                                       Visu.zcuutVariance)
                # wid.Xview.setImage(image=GPR.Xcut_pred.reshape((Visu.resY, Visu.resZ)), levels=(0, 2))
                # wid.Xview.axis.imshow(GPR.Xcut_pred.reshape((Visu.resY, Visu.resZ)),cmap=plt.get_cmap('RdYlGn'))
                wid.XimgVar.set_data(GPR.Xcut_predVariance.reshape((Visu.resY, Visu.resZ)))
                wid.XimgVar.set_clim(vmin=min(GPR.Xcut_predVariance), vmax=max(GPR.Xcut_predVariance))
                wid.XviewVar.canvas.draw_idle()

        def changeValueYVar(value):
            if self.counter:
                self.labelYcutVariance.setText("<html><head/><body><p>y = %0.1f</p></body></html>" % Visu.Y[value])
                Visu.newsetCutVariance('y', value)
                # GPR.WindSecViewGPR('y', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuut, Visu.ycuut, Visu.zcuut)
                GPR.VarianceSecViewGPR('y', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuutVariance, Visu.ycuutVariance,
                                       Visu.zcuutVariance)
                # wid.Yview.setImage(image=GPR.Ycut_pred.reshape((Visu.resZ, Visu.resX)), levels=(0, 2))
                # wid.Yview.axis.imshow(GPR.Ycut_pred.reshape((Visu.resZ, Visu.resX)),cmap=plt.get_cmap('RdYlGn'))
                wid.YimgVar.set_data(GPR.Ycut_predVariance.reshape((Visu.resZ, Visu.resX)))
                wid.YimgVar.set_clim(vmin=min(GPR.Ycut_predVariance), vmax=max(GPR.Ycut_predVariance))
                wid.YviewVar.canvas.draw_idle()

        def changeValueZVar(value):
            if self.counter:
                self.labelZcutVariance.setText("<html><head/><body><p>z = %0.1f</p></body></html>" % Visu.Z[value])
                Visu.newsetCutVariance('z', value)
                # GPR.WindSecViewGPR('z', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuut, Visu.ycuut, Visu.zcuut)
                GPR.VarianceSecViewGPR('z', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuutVariance, Visu.ycuutVariance,
                                       Visu.zcuutVariance)
                # wid.Zview.setImage(image=GPR.Zcut_pred.reshape((Visu.resX, Visu.resY)), levels=(0, 2))
                # wid.Zview.axis.imshow(GPR.Zcut_pred.reshape((Visu.resX, Visu.resY)),cmap=plt.get_cmap('RdYlGn'), vmin=-20, vmax=20)
                wid.ZimgVar.set_data(GPR.Zcut_predVariance.reshape((Visu.resX, Visu.resY)))
                wid.ZimgVar.set_clim(vmin=min(GPR.Zcut_predVariance), vmax=max(GPR.Zcut_predVariance))
                wid.ZviewVar.canvas.draw_idle()

        self.sliderZcutVariance.valueChanged.connect(changeValueZVar)
        self.sliderXcutVariance.valueChanged.connect(changeValueXVar)
        self.sliderYcutVariance.valueChanged.connect(changeValueYVar)

    def secondaryViewplot(self):
        self.Xview = MatplotlibWidget()
        self.Yview = MatplotlibWidget()
        self.Zview = MatplotlibWidget()

        # self.a = self.Xview.axis.imshow(GPR.Xcut_pred.reshape((Visu.resY, Visu.resZ)), cmap=plt.get_cmap('RdYlGn'))
        self.Ximg = self.Xview.axis.imshow(np.zeros((Visu.resY, Visu.resZ)), cmap=plt.get_cmap('coolwarm'), vmin=0,
                                           vmax=1, interpolation='bilinear')
        self.Yimg = self.Yview.axis.imshow(np.zeros((Visu.resZ, Visu.resX)), cmap=plt.get_cmap('coolwarm'), vmin=0,
                                           vmax=1, interpolation='bilinear')
        self.Zimg = self.Zview.axis.imshow(np.zeros((Visu.resX, Visu.resY)), cmap=plt.get_cmap('coolwarm'), vmin=0,
                                           vmax=1, interpolation='bilinear')

        self.XimgColorbar = self.Xview.figure.colorbar(self.Ximg)
        self.YimgColorbar = self.Yview.figure.colorbar(self.Yimg)
        self.ZimgColorbar = self.Zview.figure.colorbar(self.Zimg)

        self.layoutXview = QVBoxLayout(self.Xplotview)
        self.layoutYview = QVBoxLayout(self.Yplotview)
        self.layoutZview = QVBoxLayout(self.Zplotview)

        self.layoutXview.addWidget(self.Xview)
        self.layoutYview.addWidget(self.Yview)
        self.layoutZview.addWidget(self.Zview)

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
                # wid.Xview.setImage(image=GPR.Xcut_pred.reshape((Visu.resY, Visu.resZ)), levels=(0, 2))
                # wid.Xview.axis.imshow(GPR.Xcut_pred.reshape((Visu.resY, Visu.resZ)),cmap=plt.get_cmap('RdYlGn'))
                wid.Ximg.set_data(GPR.Xcut_pred.reshape((Visu.resY, Visu.resZ)))
                wid.Ximg.set_clim(vmin=min(GPR.Xcut_pred), vmax=max(GPR.Xcut_pred))
                wid.Xview.canvas.draw_idle()

        def changeValueY(value):
            if self.counter:
                # realValue = value / 10
                # self.labelYcut.setText("<html><head/><body><p>y = %0.1f</p></body></html>" % realValue)
                # Visu.setCut('y', realValue)

                self.labelYcut.setText("<html><head/><body><p>y = %0.1f</p></body></html>" % Visu.Y[value])
                Visu.newsetCut('y', value)
                GPR.WindSecViewGPR('y', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuut, Visu.ycuut, Visu.zcuut)
                # wid.Yview.setImage(image=GPR.Ycut_pred.reshape((Visu.resZ, Visu.resX)), levels=(0, 2))
                # wid.Yview.axis.imshow(GPR.Ycut_pred.reshape((Visu.resZ, Visu.resX)),cmap=plt.get_cmap('RdYlGn'))
                wid.Yimg.set_data(GPR.Ycut_pred.reshape((Visu.resZ, Visu.resX)))
                wid.Yimg.set_clim(vmin=min(GPR.Ycut_pred), vmax=max(GPR.Ycut_pred))
                wid.Yview.canvas.draw_idle()

        def changeValueZ(value):
            if self.counter:
                # realValue = value / 10
                # self.labelZcut.setText("<html><head/><body><p>z = %0.1f</p></body></html>" % realValue)
                # Visu.setCut('z', realValue)

                self.labelZcut.setText("<html><head/><body><p>z = %0.1f</p></body></html>" % Visu.Z[value])
                Visu.newsetCut('z', value)
                GPR.WindSecViewGPR('z', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuut, Visu.ycuut, Visu.zcuut)
                # wid.Zview.setImage(image=GPR.Zcut_pred.reshape((Visu.resX, Visu.resY)), levels=(0, 2))
                # wid.Zview.axis.imshow(GPR.Zcut_pred.reshape((Visu.resX, Visu.resY)),cmap=plt.get_cmap('RdYlGn'), vmin=-20, vmax=20)
                wid.Zimg.set_data(GPR.Zcut_pred.reshape((Visu.resX, Visu.resY)))
                wid.Zimg.set_clim(vmin=min(GPR.Zcut_pred), vmax=max(GPR.Zcut_pred))
                wid.Zview.canvas.draw_idle()

        self.sliderZcut.valueChanged.connect(changeValueZ)
        self.sliderXcut.valueChanged.connect(changeValueX)
        self.sliderYcut.valueChanged.connect(changeValueY)

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
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        try:
            if dev == 0:
                OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
                smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
                mainPos = OptiData[1:4]
                quaternion = OptiData[4:8]

                """Il s'agit de la partie SmartProbe"""
                SPP2prim = SmartProbeVect(quaternion)
                SPP2 = tuple([mainPos[0] + SPP2prim[0], mainPos[1] + SPP2prim[1], mainPos[2] + SPP2prim[2]])

                SPpos = np.array([mainPos, SPP2])
                # self.SmartProbe.setData(pos=SPpos)

                # ############################################################
                """Il s'agit de la partie du Vecteur vent avec le bout du vecteur"""
                WindP2prim = WindVect(quaternion, smartProbeData)
                WindP2 = tuple([mainPos[0] + WindP2prim[0], mainPos[1] + WindP2prim[1], mainPos[2] + WindP2prim[2]])
                Windpos = np.array([mainPos, WindP2])
                # self.Wind.setData(pos=Windpos)

                WindDotpos = np.array([WindP2])
                # self.WindVectDot.setData(pos=WindDotpos)
                ############################################################
                """Il s'agit de la partie position temps reel"""
                self.xyzPosition.setText(
                    "<html><head/><body><p>x=%0.001f</p><p>y=%0.001f</p><p>z=%0.001f</p></body></html>"
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

            if dev == 1:
                OptiData = GetOptiTrackData(self.natnet.rigidBodyList, wid.SPid)
                smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
                mainPos = OptiData[0]
                quaternion = OptiData[1]

                """Il s'agit de la partie SmartProbe"""
                SPP2prim = SmartProbeVect(quaternion)
                SPP2 = tuple([mainPos[0] + SPP2prim[0], mainPos[1] + SPP2prim[1], mainPos[2] + SPP2prim[2]])

                SPpos = np.array([mainPos, SPP2])
                self.SmartProbe.setData(pos=SPpos)

                # ############################################################
                """Il s'agit de la partie du Vecteur vent avec le bout du vecteur"""
                WindP2prim = WindVect(quaternion, smartProbeData)
                WindP2 = tuple([mainPos[0] + WindP2prim[0], mainPos[1] + WindP2prim[1], mainPos[2] + WindP2prim[2]])
                Windpos = np.array([mainPos, WindP2])
                self.Wind.setData(pos=Windpos)

                WindDotpos = np.array([WindP2])
                self.WindVectDot.setData(pos=WindDotpos)
                ############################################################
                """Il s'agit de la partie position temps reel"""
                self.xyzPosition.setText(
                    "<html><head/><body><p>x=%0.001f</p><p>y=%0.001f</p><p>z=%0.001f</p></body></html>"
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

            if dev == 2:
                OptiData = GetOptiTrackData(self.natnet.rigidBodyList, wid.SPid)
                smartProbeData = self.serial.smartprobeData
                mainPos = OptiData[0]
                quaternion = OptiData[1]

                """Il s'agit de la partie SmartProbe"""
                SPP2prim = SmartProbeVect(quaternion)
                SPP2 = tuple([mainPos[0] + SPP2prim[0], mainPos[1] + SPP2prim[1], mainPos[2] + SPP2prim[2]])

                SPpos = np.array([mainPos, SPP2])
                self.SmartProbe.setData(pos=SPpos)

                # ############################################################
                """Il s'agit de la partie du Vecteur vent avec le bout du vecteur"""
                WindP2prim = WindVect(quaternion, smartProbeData)
                WindP2 = tuple([mainPos[0] + WindP2prim[0], mainPos[1] + WindP2prim[1], mainPos[2] + WindP2prim[2]])
                Windpos = np.array([mainPos, WindP2])
                self.Wind.setData(pos=Windpos)

                WindDotpos = np.array([WindP2])
                self.WindVectDot.setData(pos=WindDotpos)
                ############################################################
                """Il s'agit de la partie position temps reel"""
                self.xyzPosition.setText(
                    "<html><head/><body><p>x=%0.001f</p><p>y=%0.001f</p><p>z=%0.001f</p></body></html>"
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

            # ############################################################
            # """Il s'agit de la partie SmartProbe"""
            # SPP2prim = SmartProbeVect(quaternion)
            # SPP2 = tuple([mainPos[0] + SPP2prim[0], mainPos[1] + SPP2prim[1], mainPos[2] + SPP2prim[2]])
            #
            # SPpos = np.array([mainPos, SPP2])
            # self.SmartProbe.setData(pos=SPpos)
            #
            # # ############################################################
            # """Il s'agit de la partie du Vecteur vent avec le bout du vecteur"""
            # WindP2prim = WindVect(quaternion, smartProbeData)
            # WindP2 = tuple([mainPos[0] + WindP2prim[0], mainPos[1] + WindP2prim[1], mainPos[2] + WindP2prim[2]])
            # Windpos = np.array([mainPos, WindP2])
            # self.Wind.setData(pos=Windpos)
            #
            # WindDotpos = np.array([WindP2])
            # self.WindVectDot.setData(pos=WindDotpos)
            # ############################################################
            # """Il s'agit de la partie position temps reel"""
            # self.xyzPosition.setText("<html><head/><body><p>x=%0.001f</p><p>y=%0.001f</p><p>z=%0.001f</p></body></html>"
            #                          % (mainPos[0], mainPos[1], mainPos[2]))
            # self.SpeedLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[0]))
            # # self.angleofattackLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[2]))
            # self.pitchangleLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[2]))
            # # self.sideslipeLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[3]))
            # self.SpeedLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[0]))
            # self.angleofattackLabelValue.setText(
            #     "<html><head/><body>%0.001f</body></html>" % (convert_to_alpha(SPP2prim, WindP2prim)))
            # self.pitchangleLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (pitch_angle(SPP2prim)))
            # self.sideslipeLabelValue.setText(
            #     "<html><head/><body>%0.001f</body></html>" % (convert_to_beta(SPP2prim, WindP2prim)))

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
                    L = np.array(tail('testdata.csv', 1000))
                    OptiData = L[wid.c]
                    wid.c += 1
                    # OptiData = np.array(tail('Data/OptiTrackData.csv', 1)[0])
                    smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
                    mainPos = OptiData[1:4]
                    quaternion = OptiData[4:8]

                    """Il s'agit de la partie SmartProbe"""
                    SPP2prim = SmartProbeVect(quaternion)
                    SPP2 = tuple([mainPos[0] + SPP2prim[0], mainPos[1] + SPP2prim[1], mainPos[2] + SPP2prim[2]])

                    SPpos = np.array([mainPos, SPP2])
                    wid.SmartProbe.setData(pos=SPpos)

                    ############################################################
                    """Il s'agit de la partie du Vecteur vent avec le bout du vecteur"""
                    # WindP2prim = WindVect(quaternion, smartProbeData)
                    WindP2prim = fonctionvent(mainPos[0], mainPos[1], mainPos[2])
                    WindP2 = tuple([mainPos[0] + WindP2prim[0], mainPos[1] + WindP2prim[1], mainPos[2] + WindP2prim[2]])
                    Windpos = np.array([mainPos, WindP2])
                    wid.Wind.setData(pos=Windpos)

                    WindDotpos = np.array([WindP2])
                    wid.WindVectDot.setData(pos=WindDotpos)

                    """Il s'agit de la partie position temps reel"""
                    wid.xyzPosition.setText(
                        "<html><head/><body><p>x=%0.001f</p><p>y=%0.001f</p><p>z=%0.001f</p></body></html>"
                        % (mainPos[0], mainPos[1], mainPos[2]))
                    wid.SpeedLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[0]))
                    # self.angleofattackLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[2]))
                    wid.pitchangleLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[2]))
                    # self.sideslipeLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[3]))
                    wid.SpeedLabelValue.setText("<html><head/><body>%0.001f</body></html>" % (smartProbeData[0]))
                    wid.angleofattackLabelValue.setText(
                        "<html><head/><body>%0.001f</body></html>" % (convert_to_alpha(SPP2prim, WindP2prim)))
                    wid.pitchangleLabelValue.setText(
                        "<html><head/><body>%0.001f</body></html>" % (pitch_angle(SPP2prim)))
                    wid.sideslipeLabelValue.setText(
                        "<html><head/><body>%0.001f</body></html>" % (convert_to_beta(SPP2prim, WindP2prim)))

                    Track.TrackStorage(mainPos, WindP2prim)

                if dev == 1:
                    OptiData = GetOptiTrackData(wid.natnet.rigidBodyList, wid.SPid)
                    smartProbeData = np.array(tail('Data/SmartProbeData.csv', 1)[0])
                    mainPos = OptiData[0]
                    quaternion = OptiData[1]
                    Track.TrackStorage(mainPos, WindVect(quaternion, smartProbeData))

                if dev == 2:
                    OptiData = GetOptiTrackData(wid.natnet.rigidBodyList, wid.SPid)
                    smartProbeData = wid.serial.smartprobeData
                    mainPos = OptiData[0]
                    quaternion = OptiData[1]
                    Track.TrackStorage(mainPos, WindVect(quaternion, smartProbeData))

                GPR.setDataForGPR(Track.wholeTrack, Track.wholeWindTrack)
                GPR.predictWindGPR()
                pos = np.empty((0, 3))
                color = np.empty((0, 4))
                a = Visu.Points
                b = Visu.windDistribution(GPR.Yx_pred / 2, GPR.Yy_pred / 2, GPR.Yz_pred / 2)

                for i in range(Visu.nbpoint):
                    pos = np.concatenate((pos, a[i].reshape((1, 3)), b[i].reshape(1, 3)))
                    var = GPR.Variance(i)
                    col = np.array(glColor(VarianceToColor(var)))
                    color = np.concatenate((color, col.reshape(1, 4), col.reshape(1, 4)))

                wid.WindDistribution.setData(pos=pos, color=color)

                GPR.WindSecViewGPR('all', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuut, Visu.ycuut, Visu.zcuut)
                wid.Ximg.set_data(GPR.Xcut_pred.reshape((Visu.resY, Visu.resZ)))
                wid.Yimg.set_data(GPR.Ycut_pred.reshape((Visu.resZ, Visu.resX)))
                wid.Zimg.set_data(GPR.Zcut_pred.reshape((Visu.resX, Visu.resY)))
                wid.Ximg.set_clim(vmin=min(GPR.Xcut_pred), vmax=max(GPR.Xcut_pred))
                wid.Yimg.set_clim(vmin=min(GPR.Ycut_pred), vmax=max(GPR.Ycut_pred))
                wid.Zimg.set_clim(vmin=min(GPR.Zcut_pred), vmax=max(GPR.Zcut_pred))
                wid.Xview.canvas.draw_idle()
                wid.Yview.canvas.draw_idle()
                wid.Zview.canvas.draw_idle()

                GPR.VarianceSecViewGPR('all', Visu.resX, Visu.resY, Visu.resZ, Visu.xcuutVariance, Visu.ycuutVariance,
                                       Visu.zcuutVariance)
                wid.XimgVar.set_data(GPR.Xcut_predVariance.reshape((Visu.resY, Visu.resZ)))
                wid.YimgVar.set_data(GPR.Ycut_predVariance.reshape((Visu.resZ, Visu.resX)))
                wid.ZimgVar.set_data(GPR.Zcut_predVariance.reshape((Visu.resX, Visu.resY)))
                wid.XimgVar.set_clim(vmin=min(GPR.Xcut_predVariance), vmax=max(GPR.Xcut_predVariance))
                wid.YimgVar.set_clim(vmin=min(GPR.Ycut_predVariance), vmax=max(GPR.Ycut_predVariance))
                wid.ZimgVar.set_clim(vmin=min(GPR.Zcut_predVariance), vmax=max(GPR.Zcut_predVariance))
                wid.XviewVar.canvas.draw_idle()
                wid.YviewVar.canvas.draw_idle()
                wid.ZviewVar.canvas.draw_idle()
                # wid.YimgColorbar.clim(vmin=min(GPR.Ycut_pred), vmax=max(GPR.Ycut_pred))
            except (NameError, TypeError):
                continue

    def stop(self):
        self.isRunning = False


class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self):
        super(MatplotlibWidget, self).__init__()
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axis = self.figure.add_subplot(111)

        self.layoutvertical = QVBoxLayout(self)
        self.layoutvertical.addWidget(self.canvas)


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
