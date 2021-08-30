from time import sleep
from pyqtgraph.Qt import QtWidgets
from WEGPR import app_1
import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wid = app_1()
    sleep(2)
    wid.show()
    sys.exit(app.exec_())
