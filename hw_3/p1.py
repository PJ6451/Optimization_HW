import numpy as np
import conj_grad as scg
import matplotlib.pyplot as plt

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

n_list = [5,8,12,20]
iter_list = []
cond_list = []
eig_5_list = []
eig_8_list = []
eig_12_list = []
eig_20_list = []

for n in n_list:
    A = np.zeros([n,n])
    for i in range(1,n+1):
        for j in range(1,n+1):
            A[i-1,j-1] = 1/(i+j-1)
    
    evals, evecs = np.linalg.eig(A)
    if n == 5:
        eig_5_list = np.log10(np.abs(evals))
    elif n == 8:
        eig_8_list = np.log10(np.abs(evals))
    elif n == 12:
        eig_12_list = np.log10(np.abs(evals))
    else:
        eig_20_list = np.log10(np.abs(evals))

    cond_list.append(np.log10(np.linalg.cond(A)))
    b = np.ones([n,1])
    x_0 = np.zeros([n,1])

    r_list = scg.standard_cg(x_0, A, b)
    r_array = np.array(r_list)
    iter_list.append(r_array.size-1)
    plot_1_a(n,r_array)

plot_1_b(n_list, iter_list)
plot_1_c(n_list, cond_list)
plot_1_d(eig_5_list, eig_8_list, eig_12_list, eig_20_list)