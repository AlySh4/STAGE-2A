import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# Set dimension.
d = 1
# Number of training points.
n = 100
# Length of the training set.
L = 2
# Generate training features.
x = np.linspace(start=0, stop=L, num=n)
X = x.reshape(n, d)

print(X)
print(X[1])
print(x)
print(x[1])

sigma_n = 0.4
# Errors. #il s'agit de l'epsilon dans la formule y = f(x) +epsilon
epsilon = np.random.normal(loc=0, scale=sigma_n, size=n)
print("epsilon", epsilon)


# Generate non-linear function.
def f(x):
    f = np.sin((4 * np.pi) * x) + np.sin((7 * np.pi) * x) + np.sin((3 * np.pi) * x)
    return f


f_x = f(x)

# Observed target variable.
y = f_x + epsilon

print("prout", y)

n_star = n
x_star = np.linspace(start=0, stop=(L + 0.5), num=n_star)

X_star = x_star.reshape(n_star, d)
############################################################################


# Define kernel parameters. # ce sont les parametres principaux pour construire notre processus gaussien
l = 0.1
sigma_f = 2

# Define kernel object. # il s'agit de la fonction de covariance: la fonction avec laquelle nous allons definir notre
# processus gaussien
kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds=(1e-2, 1e2)) \
         * RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2))

# Define GaussianProcessRegressor object.
gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n ** 2, n_restarts_optimizer=10, )

# Fit to data using Maximum Likelihood Estimation of the parameters.
gp.fit(X, y)

y_pred = gp.predict(X_star)

#############################################################################
fig, ax = plt.subplots(figsize=(15, 8))
# Plot training data.
sns.scatterplot(x=x, y=y, label='training data', ax=ax)
# Plot "true" linear fit.
sns.lineplot(
    x=x_star,
    y=f(x_star),
    color='red',
    label='f(x)',
    ax=ax
)
# Plot prediction
sns.lineplot(x=x_star, y=y_pred, color='green', label='pred')
ax.set(title='Prediction & Credible Interval')
ax.legend(loc='lower left')

plt.show()
