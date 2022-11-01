import numpy as np
import conj_grad as scg
import matplotlib.pyplot as plt

def plot_3(r_array_1: np.ndarray, r_array_2: np.ndarray):
    iters = np.arange(0,r_array_1.size)
    plt.plot(iters, r_array_1, label = 'clustered eigenvalues')
    plt.plot(iters, r_array_2[iters], label = 'uniformly distributed eigenvalues')
    plt.title("$\log_{10} ||x - x^*||$ for different eigenvalue clusterings")
    plt.xlabel("Iterations")
    plt.ylabel("$\log_{10} ||x - x^*||$")
    plt.legend()
    filename = "scg_3"
    plt.savefig(filename)
    plt.clf()

n = 20
b = np.ones(n)
x_0 = np.zeros(n)

k = np.append(
    np.array([95, 100, 115, 150, 125]),
    np.arange(.95,1.05,((1.05-.95)/15)),
    0
)

A = np.diag(k)
x_star = scg.standard_cg_get_min(x_0, A, b)
r_list_1 = scg.standard_cg_op_norm(x_star, x_0, A ,b)
r_array_1 = np.array(r_list_1)

k = np.random.uniform(0, 100, n)
A = np.diag(k)
x_star = scg.standard_cg_get_min(x_0, A, b)
r_list_2 = scg.standard_cg_op_norm(x_star, x_0, A ,b)
r_array_2 = np.array(r_list_2)

plot_3(r_array_1, r_array_2)