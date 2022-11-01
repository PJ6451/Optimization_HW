import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]), 200*(x[1]-x[0]**2)])

def hess_f(x):
    return np.array([[-400*x[1] + 1200*x[0]**2 + 2,-400*x[0]],[-400*x[0],200]])

def cg_core(A: np.ndarray, b: np.ndarray, nu_k: float):
    j = 0
    x_k = np.zeros(2)
    r_k = A @ x_k - b
    p_0 = -r_k
    p_k = -r_k
    while np.linalg.norm(r_k,2) > nu_k * np.linalg.norm(r_k,2):
        rTr = r_k.T @ r_k
        Apk = A @ p_k
        pTApk = p_k.T @ Apk
        alpha_k = rTr / pTApk
        if pTApk <= 0:
            if j == 0:
                p_k = p_0
                print("yes")
            else:
                p_k = x_k
        x_k = x_k + alpha_k * p_k
        r_k_1 = r_k + alpha_k * Apk
        beta_k = (r_k_1.T @ r_k_1) / (rTr)
        p_k = -r_k_1 + beta_k * p_k
        r_k = r_k_1
        j += 1
    return x_k
    
def line_search_newton(x_k: np.ndarray):
    k = 0
    x_list = [np.log10(f(x_k))]
    grad_f_k = grad_f(x_k)
    norm_grad_f = np.linalg.norm(grad_f_k,2)
    while np.linalg.norm(grad_f_k) >= 10**(-8):
        A = hess_f(x_k)
        b = -grad_f_k
        nu_k = min(10**-3, np.sqrt(norm_grad_f))

        p_k = cg_core(A, b, nu_k)
        alpha_k = linesearch_armijo(p_k, x_k)
        
        x_k = x_k + alpha_k * p_k
        grad_f_k = grad_f(x_k)
        norm_grad_f = np.linalg.norm(grad_f_k,2)
        
        k += 1
        x_list.append(np.log10(f(x_k)))
    return x_list

def linesearch_armijo(p_k: np.ndarray, x_k: np.ndarray):
    alpha = 1
    rho = 0.5
    c = 10**(-4)
    while armijo(x_k, alpha, c, p_k):
        alpha = alpha*rho
    return alpha

def armijo(x_k, alpha, c, p_k):
    return f(x_k + alpha*p_k) > f(x_k) + (c*alpha*p_k.T) @ grad_f(x_k)

def plot_4(x_k: np.ndarray, x_list: np.ndarray):
    iters = np.arange(1,x_list.size + 1)
    plt.plot(iters, x_list)
    plt.title('$\log_{10} (f(x_k))$ for $x_0$ = (' + str(x_k[0]) + ', ' + str(x_k[1]) + ')')
    plt.xlabel("Iterations")
    plt.ylabel("$\log_{10} (f(x_k))$")
    filename = "scg_4_x_0_1" 
    plt.savefig(filename)
    plt.clf()

x_k = np.array([-1.2,1])
x_list = line_search_newton(x_k)
x_array = np.array(x_list)
plot_4(x_k, x_array)