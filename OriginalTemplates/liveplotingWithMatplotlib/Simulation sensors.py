import random
import sys
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as an
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget

plt.style.use('fivethirtyeight')

index = count()

fig, ax = plt.subplots()


def animate(i):
    data = pd.read_csv('data.csv')
    x = data['x_value']
    y1 = data['total_1']
    y2 = data['total_2']

    plt.cla()

    plt.plot(x, y1, label='Channel 1')
    plt.plot(x, y2, label='Channel 2')

    plt.legend(loc='upper left')


ani = an.FuncAnimation(fig, func=animate, interval=1)

app = QApplication(sys.argv)

window = QWidget()

plt.tight_layout()
plt.show()
