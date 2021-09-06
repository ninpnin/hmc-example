import numpy as np
from matplotlib import pyplot as plt
import seaborn as sbn
import tensorflow as tf
import progressbar
import pandas as pd

@tf.function
def loss(theta):
    diff1 = theta - 0.25
    diff2 = theta + 0.5
    norm1 = tf.linalg.norm(diff1)
    norm2 = tf.linalg.norm(diff2)
    f = 0.3 * tf.math.exp(-norm1) + tf.math.exp(-norm2)
    return tf.math.log(f)

def leapfrog_step(theta_t, r_t, eta):
    with tf.GradientTape() as tape:
        y = loss(theta_t)

    grad1 = tape.gradient(y, theta_t)
    r_t.assign(r_t.numpy() + grad1.numpy() * eta * 0.5)

    theta_t.assign(theta_t.numpy() + r_t.numpy() * eta)

    with tf.GradientTape() as tape:
        y = loss(theta_t)

    grad2 = tape.gradient(y, theta_t)
    r_t.assign(r_t.numpy() + grad2.numpy() * eta * 0.5)

    return theta_t, r_t

def hmc_step(theta_old, theta, theta_t, r_0, r_old, r_t, eta, L=10):
    r_0.assign(np.random.normal(size=theta.shape))
    r_t.assign(r_0.numpy())
    theta_t.assign(theta)

    for l in range(L):
        x, r = leapfrog_step(theta_t, r_t, eta)

    loss1 = loss(theta_t)
    loss0 = loss(theta)

    numerator = tf.math.exp(loss1 - tf.reduce_sum(tf.multiply(r_t,r_t))) 
    denominator = tf.math.exp(loss0 - tf.reduce_sum(tf.multiply(r_0,r_0))) 
    
    ratio = max(1.0, numerator/denominator)

    if np.random.rand() <= ratio:
        return theta_t.numpy(), True
    else:
        return theta.numpy(), False

def hmc_sampling(M):
    theta_old = tf.Variable(np.random.normal(size=(2,)))
    theta = tf.Variable(np.zeros(theta_old.shape))
    theta_t = tf.Variable(np.zeros(theta_old.shape))
    r_0 = tf.Variable(np.zeros(theta_old.shape))
    r_old = tf.Variable(np.zeros(theta_old.shape))
    r_t = tf.Variable(np.zeros(theta_old.shape))
    eta = 0.1
    rows = []
    for m in progressbar.progressbar(range(M)):
        sample, accepted = hmc_step(theta_old, theta, theta_t, r_0, r_old, r_t, eta)
        rows.append(list(sample) + [accepted])

    shape = theta.shape[0]
    columns = ["x_" + str(i) for i in range(shape)] + ["accepted"]
    return pd.DataFrame(rows, columns=columns)

def main():
    N = 1000
    df = hmc_sampling(N)

    burn_in = 0#1000
    samples = df.tail(N - burn_in)
    print(samples)

if __name__ == '__main__':
    main()