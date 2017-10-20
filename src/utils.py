
import numpy as np


np.random.seed(46)

# generating synthetic data with random projection
def gen_synthetic_data(d, prj_d, n):
    z = np.random.normal(loc=0., scale=1.0, size=(n, d))
    r_x = np.random.uniform(-1, 1, size=(d, prj_d))
    r_y = np.random.uniform(-1, 1, size=(d, prj_d))
    x = z.dot(r_x) + np.random.randn(n, prj_d)  # true_x + gaussian noise
    y = z.dot(r_y) + np.random.randn(n, prj_d)
    return x.astype(np.float32), y.astype(np.float32)
