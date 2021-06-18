import pandas as pd
import csv
import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

sigma_n = 0.4
m = 50

l = 0.1
sigma_f = 1

kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds=(1e-2, 1e2)) \
         * RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2))

gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n ** 2, n_restarts_optimizer=10, )

fieldnames = ["x", "ypred", "f(x)"]

with open('dataGpr.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
k = 0

while True:
    while True:
        data = np.loadtxt('data.csv', delimiter=',', skiprows=1)
        x = data[m * k:m * (k + 1), 0]
        f = data[m * k:m * (k + 1), 1]
        y = data[m * k:m * (k + 1), 2]
        if len(x) == m and len(y) == m:
            break

    Xm = x.reshape((m, 1))
    gp.fit(Xm, y)
    y_pred = gp.predict(Xm)

    with open('dataGpr.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for i in range(0, m):
            info = {"x": Xm[i][0], "ypred": y_pred[i], "f(x)": f[i]}
            csv_writer.writerow(info)

    k += 1
