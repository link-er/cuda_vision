# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def S0(size=100):
    x = np.arange(size)
    y = np.random.normal(0, 1, size=size)
    return x, y

x, y = S0()
plt.scatter(x, y);

# <codecell>

def S1(size=100):
    x = np.arange(size)
    y = 2*x + 3
    return x, y

x, y = S1()
plt.scatter(x, y);

# <codecell>

def S01(size=100, noise_level=1):
    x = np.arange(size)
    y = 2*x + 3
    dataset = y + np.random.normal(0, noise_level, size=size)
    return x, y, dataset

x, y, dataset = S01()
plt.scatter(x, dataset, label='y')
plt.plot(x, y, 'r', label='ground truth')
plt.legend()

# <codecell>

def CD(n_classes=3, size=(100, 2)):
    means = np.random.uniform(size=(n_classes, size[1]))
    stds = np.random.uniform(0, 0.2, size=n_classes)
    X = [np.random.normal(m, s, size=size) for m, s in zip(means,stds)]
    y = [[i] * size[0] for i in xrange(n_classes)]
    X = np.vstack(X)
    y = np.ravel(y)
    return X, y

np.random.seed(1)
X, y = CD()
plt.scatter(X[:,0], X[:,1], c=y)
plt.gca().set_aspect(1.)
np.savetxt('tricky.dat', X)

# <codecell>


# <codecell>


