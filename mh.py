"""
Basic random walk Metropolis-Hastings
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sbn
import tensorflow as tf
import pandas as pd
import progressbar

@tf.function
def loss(theta):
    diff1 = theta - 5.0
    diff2 = theta 
    norm1 = - tf.linalg.norm(diff1) * 2.0
    norm2 = - tf.linalg.norm(diff2) * 2.0
    f = 0.3 * tf.math.exp(norm1) + tf.math.exp(norm2)
    return tf.math.log(f)

@tf.function
def mh_step(x_old):
    x_new = tf.random.normal(x_old.shape) + x_old    
    loss_ratio = loss(x_new) - loss(x_old)
    loss_ratio = tf.math.exp(loss_ratio)
    if tf.random.uniform(()) <= loss_ratio:
        return x_new
    else:
        return x_old

def mh_sampling(steps, D):
    x = tf.random.normal((D,))
    rows = []
    for s in progressbar.progressbar(range(steps)):
        rows.append(x.numpy())
        x = mh_step(x)

    shape = x.shape[0]
    columns = ["x_" + str(i) for i in range(shape)]
    return pd.DataFrame(rows, columns=columns)

def main():
    print("Let's sample some stuff")

    D = 8
    N = 15000
    df = mh_sampling(N, D)
    print(df)

    ## Discard burn-in samples
    burn_in = min(N // 2, max(1000, N // 10))
    samples = df.tail(N- burn_in)
    print(samples)

    g = sbn.histplot(samples["x_0"], bins=40)
    plt.show()

if __name__ == '__main__':
    main()