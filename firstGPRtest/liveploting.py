import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as an

fig, ax = plt.subplots()


def animate(i):
    data = pd.read_csv('data.csv')
    x = data['x']
    y1 = data['f(x)']
    y2 = data['Y']

    plt.cla()

    plt.plot(x, y1, label='f(x)', c='r')
    plt.scatter(x, y2, label='Y = f(x) + epsilon')

    plt.legend(loc='upper left')


ani = an.FuncAnimation(fig, func=animate, interval=1)

plt.show()
