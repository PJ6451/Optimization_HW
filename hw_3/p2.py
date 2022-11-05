import numpy as np
import conj_grad as scg
import matplotlib.pyplot as plt

def plot_2(n: int, r_array: np.ndarray):
    iters = np.arange(1,r_array.size + 1)
    plt.plot(iters, r_array)
    plt.title(r'$\log_{10} ||r_k||$ for n = ' + str(n))
    plt.xlabel("Iterations")
    plt.ylabel("$\log_{10} ||r_k||$")
    plt.xticks(range(0,20,4))
    filename = "scg_2"
    plt.savefig(filename)
    plt.clf()

def Hilbert(n):
    i, j = np.ogrid[1:n+1, 1:n+1]
    return 1/(i + j - 1)

n= 20
A = Hilbert(n)

b = np.ones(n)
x_0 = np.zeros(n)

r_list = scg.conj_dir_eig(n, x_0, A, b)
r_array = np.array(r_list)
plot_2(n,r_array)