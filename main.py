from pyqtgraph.Qt import QtWidgets
from design import app_1
import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wid = app_1()
    wid.show()
    sys.exit(app.exec_())
