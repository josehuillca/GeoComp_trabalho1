import numpy as np

np.random.seed(21)

def generate_norm_distrib(nPoints: int, mean: int = 0, std: int = 1) {
    xSample = np.random.normal(mean, std, n)
    ySample = np.random.normal(mean, std, n)

    return list(zip(xSample, ySample))
}