import csv
import time
import numpy as np


# Generate non-linear function qu'il faut trouver plus tard à l'aide GPR
def f(x):
    f = np.sin((4 * np.pi) * x) + np.sin((7 * np.pi) * x) + np.sin((3 * np.pi) * x)
    return f


def f1(x):
    return x


def genY(f):
    # Set dimension.
    d = 1
    # Number of training points.
    n = 10000
    # Length of the training set.
    L = 10
    # Generate training features.
    x = np.linspace(start=0, stop=L, num=n)
    sigma_n = 0.4

    # Errors. #il s'agit de l'epsilon dans la formule y = f(x) +epsilon
    epsilon = np.random.normal(loc=0, scale=sigma_n, size=n)
    f_x = f(x)
    # Observed target variable.
    y = f_x + epsilon

    fieldnames = ["x", "f(x)", "Y"]

    with open('data.csv', 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    i = 0
    while True:
        tps1 = time.clock()
        with open('data.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            info = {
                "x": x[i],
                "f(x)": f(x[i]),
                "Y": y[i]
            }
            csv_writer.writerow(info)
            # print(x[i], f(x[i]), y[i])
            i += 1
        time.sleep(0.001)
        tps2 = time.clock()
        print(tps2 - tps1)


genY(f)
