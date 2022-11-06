import matplotlib.pyplot as plt
import numpy as np

def plot_1_a(n: int, r_array: np.ndarray):
    iters = np.arange(1,r_array.size + 1)
    plt.plot(iters, r_array)
    plt.title(r'$\log_{10} ||r_k||$ for n = ' + str(n))
    plt.xlabel("Iterations")
    plt.ylabel("$\log_{10} ||r_k||$")
    filename = "scg_1_a_n" + str(n)
    plt.savefig(filename)
    plt.clf()

def plot_1_b(n_list: list, iter_list: np.ndarray):
    plt.scatter(n_list, iter_list)
    plt.title("Number of iterations per size n")
    plt.xlabel("n")
    plt.ylabel("Iterations")
    filename = "scg_1_b"
    plt.savefig(filename)
    plt.clf()

def plot_1_c(n_list: list, cond_list: np.ndarray):
    plt.scatter(n_list, cond_list)
    plt.title("$\log_{10} ||\kappa||$ per size n")
    plt.xlabel("n")
    plt.ylabel("$\log_{10} ||\kappa||$")
    filename = "scg_1_c"
    plt.savefig(filename)
    plt.clf()

def plot_1_d(eig_5_list, eig_8_list, eig_12_list, eig_20_list):
    plt.scatter(np.arange(0,5),np.sort(eig_5_list), label = 'n=5')
    plt.scatter(np.arange(0,8),np.sort(eig_8_list), label = 'n=8')
    plt.scatter(np.arange(0,12),np.sort(eig_12_list), label = 'n=12')
    plt.scatter(np.arange(0,20),np.sort(eig_20_list), label = 'n=20')
    plt.legend()
    plt.title("$\log_{10} |\lambda_n|$ per size n")
    plt.xlabel("n")
    plt.ylabel("$\log_{10} |\lambda_n|$")
    plt.xticks(range(0,20,4))  
    filename = "scg_1_d"
    plt.savefig(filename)
    plt.clf()

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

def plot_4(x_k: np.ndarray, x_list: np.ndarray):
    iters = np.arange(1,x_list.size + 1)
    plt.plot(iters, x_list)
    plt.title('$\log_{10} (f(x_k))$ for $x_0$ = (' + str(x_k[0]) + ', ' + str(x_k[1]) + ')')
    plt.xlabel("Iterations")
    plt.ylabel("$\log_{10} (f(x_k))$")
    filename = "scg_4_x_0_1" 
    plt.savefig(filename)
    plt.clf()