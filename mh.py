"""
Basic random walk Metropolis-Hastings
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sbn
import tensorflow as tf
import pandas as pd
import progressbar

def loss(theta):
    diff1 = theta - 5.
    diff2 = theta
    norm1 = - tf.linalg.norm(diff1) * 2
    norm2 = - tf.linalg.norm(diff2) * 2
    f = 0.3 * tf.math.exp(norm1) + tf.math.exp(norm2)
    return tf.math.log(f)

def mh_step(x_old):
    x_new = np.random.normal(size=x_old.shape) + x_old    
    loss_ratio = loss(x_new) - loss(x_old)
    loss_ratio = np.exp(loss_ratio)
    loss_ratio = min(1.0, loss_ratio)
    if np.random.rand() <= loss_ratio:
        return x_new
    else:
        return x_old

def mh_sampling(steps):
    x = np.random.normal(size=(2,))
    rows = []
    for s in progressbar.progressbar(range(steps)):
        rows.append(x)
        x = mh_step(x)

    shape = x.shape[0]
    columns = ["x_" + str(i) for i in range(shape)]
    return pd.DataFrame(rows, columns=columns)

def main():
    print("Let's sample some stuff")

    N = 150000
    df = mh_sampling(N)
    print(df)

    ## Discard burn-out samples
    burn_in = 1000
    samples = df.tail(N- burn_in)
    print(samples)

    g = sbn.histplot(samples["x_0"], bins=40)
    plt.show()

if __name__ == '__main__':
    main()