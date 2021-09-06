import numpy as np
from matplotlib import pyplot as plt
import seaborn as sbn
import tensorflow as tf

def loss(theta):
    diff1 = theta - 0.25
    diff2 = theta + 0.5
    norm1 = tf.linalg.norm(diff1)
    norm2 = tf.linalg.norm(diff2)
    f = 0.3 * tf.math.exp(-norm1) + tf.math.exp(-norm2)
    return - tf.math.log(f)

def leapfrog_step(theta_t, r_t, eta):
    with tf.GradientTape as tape:
        y = loss(theta)

    grad1 = tape.gradient(y, theta_t)
    r_t.assign(r.numpy() + grad1.numpy() * eta * 0.5)

    theta_t.assign(theta_t.numpy() + r.numpy() * eta)

    with tf.GradientTape as tape:
        y = loss(theta_t)

    grad2 = tape.gradient(y, theta_t)
    r_t.assign(r.numpy() + grad2.numpy() * eta * 0.5)

    return theta_t, r_t

def hmc_step(x_old, r_old):
    r_0 = np.random.normal(size=r.shape)
    x, r = x_old, r_0
    for l in range(L):
        x, r = leapfrog_step(x, r)

    return x, r

def hmc_sampling(M):
    theta_old = tf.Variable(np.random.normal(size=(2,)))
    momentum = tf.Variable(np.zeros(position.shape))
    rows = []
    for m in range(M):
        position, momentum = hmc_step(position, momentum)
        rows.append(position.numpy())

    shape = x.shape[0]
    columns = ["x_" + str(i) for i in range(shape)]
    return pd.DataFrame(rows, columns=columns)

def main():
    print("Let's sample some stuff")

    x = tf.constant([0.3,0.4])
    print(loss(x))
    print(loss(x + 0.1))
    pass

if __name__ == '__main__':
    main()