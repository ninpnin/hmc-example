"""
Sample from multivariate Gaussian mixture models using HMC
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sbn
import argparse
import tensorflow as tf
import progressbar
import pandas as pd

# Gaussian mixture model
# 3 units standard normal at 5.0
# 10 units standard normal at 0.0
@tf.function
def p(theta):
    diff1 = theta - 5.0
    diff2 = theta
    norm1 = tf.linalg.norm(diff1) * 2.0
    norm2 = tf.linalg.norm(diff2) * 2.0
    f = 0.3 * tf.math.exp(-norm1) + tf.math.exp(-norm2)
    return f

@tf.function
def log_p(theta):
    return tf.math.log(p(theta))

@tf.function
def leapfrog_step(theta_t, r_t, eta):
    with tf.GradientTape() as tape:
        y = log_p(theta_t)

    grad1 = tape.gradient(y, theta_t)
    r_t.assign_add(grad1 * eta * 0.5)

    theta_t.assign_add(r_t * eta)

    with tf.GradientTape() as tape:
        y = log_p(theta_t)

    grad2 = tape.gradient(y, theta_t)
    r_t.assign_add(grad2 * eta * 0.5)

    return theta_t, r_t

@tf.function
def hmc_step(theta, theta_t, r_0, r_old, r_t, eta, L=10):
    r_0.assign(tf.random.normal(theta.shape, dtype=tf.float64))
    r_t.assign(r_0)
    theta_t.assign(theta)

    for l in range(L):
        _ = leapfrog_step(theta_t, r_t, eta)        

    loss1 = log_p(theta_t)
    loss0 = log_p(theta)

    energy1 = - 0.5 * tf.reduce_sum(tf.multiply(r_t,r_t))
    energy0 = - 0.5 * tf.reduce_sum(tf.multiply(r_0,r_0))
    numerator = tf.math.exp(loss1 + energy1) 
    denominator = tf.math.exp(loss0 + energy0) 
    
    ratio = numerator/denominator
    random_number = tf.random.uniform((), dtype=tf.float64)
    if tf.reduce_sum(random_number) <= tf.reduce_sum(ratio):
        theta.assign(theta_t)
        return theta_t, True
    else:
        return theta, False

def hmc_sampling(M, eta=0.1, L=10, D=2):
    theta = tf.Variable(np.random.normal(size=(D,)))
    theta_t = tf.Variable(np.zeros(theta.shape))
    r_0 = tf.Variable(np.zeros(theta.shape))
    r_old = tf.Variable(np.zeros(theta.shape))
    r_t = tf.Variable(np.zeros(theta.shape))
    rows = []
    for m in progressbar.progressbar(range(M)):
        sample, accepted = hmc_step(theta, theta_t, r_0, r_old, r_t, eta, L=L)
        rows.append(list(sample.numpy()) + [bool(accepted)])

    shape = theta.shape[0]
    columns = ["x_" + str(i) for i in range(shape)] + ["accepted"]
    return pd.DataFrame(rows, columns=columns)

def main(args):
    N = args.N
    burn_in = min(N // 2, max(1000, N // 10))
    eta = args.eta
    L = args.L
    chains = args.chains

    chain_dfs = []
    for chain in range(chains):
        print("Chain", chain)
        df = hmc_sampling(N, eta=eta, L=L, D=args.D)

        chain_df = df.tail(N - burn_in)
        chain_df["chain"] = chain
        chain_dfs.append(chain_df)    

    samples = pd.concat(chain_dfs)

    chain_variances = samples.groupby("chain").var().mean()
    variances = samples.var()

    r_hat = chain_variances / variances
    print(r_hat)

    accepted = samples[samples["accepted"] == True]
    print("Accepted ratio", len(accepted) / len(samples))
    
    samples.to_csv("draws/hmc.csv", index=False)
    g = sbn.histplot(samples["x_0"], bins=40)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=5000, help="Number of samples per chain")
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--L", type=int, default=25, help="Leapfrog steps")
    parser.add_argument("--D", type=int, default=2, help="Dimensionality")
    args = parser.parse_args()

    main(args)