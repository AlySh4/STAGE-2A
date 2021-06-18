import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as an

fig, ax = plt.subplots()


def animate(i):
    data = pd.read_csv('dataGpr.csv')
    x = data['x']
    y1 = data['ypred']
    y2 = data['f(x)']

    plt.cla()

    plt.plot(x, y1, label='ypred', c='g')
    plt.plot(x, y2, label='f(x)', c='r')

    plt.legend(loc='upper left')


ani = an.FuncAnimation(fig, func=animate, interval=1)

plt.show()
