import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f(x):
    return np.array([400*x[0]**3 - 400*x[1]*x[0] + 2*x[0] - 2, 200*(x[1]-x[0]**2)])

def hess_f(x):
    return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])

def cg_core(A: np.ndarray, b: np.ndarray, nu_k: float):
    j = 0
    x_k = np.array([0,0])
    r_k = A @ x_k - b
    p_k = -r_k
    ep = nu_k * np.linalg.norm(b,2)
    while np.linalg.norm(r_k,2) > ep:
        Ap = A @ p_k
        pAp = p_k.T @ A @ p_k
        if pAp <= 0:
            print('Negative Curvature Detected')
            if j == 0:
                return p_k
            else:
                return x_k
        rTr = r_k.T @ r_k
        alpha_k = rTr / pAp
        x_k = x_k + alpha_k * p_k
        r_k = r_k + alpha_k * Ap
        beta_k = (r_k.T @ r_k) / (rTr)
        p_k = -r_k + beta_k * p_k
        j += 1
    return x_k
    
def line_search_newton(x_k: np.ndarray):
    k = 0
    x_list = [np.log10(f(x_k))]
    while np.linalg.norm(grad_f(x_k),2) >= 10**(-8):
        nu_k = min(10**(-3), np.sqrt(np.linalg.norm(grad_f(x_k),2)))
        p_k = cg_core(hess_f(x_k), -grad_f(x_k), nu_k)
        alpha_k = linesearch_armijo(p_k, x_k)
        x_k = x_k + alpha_k * p_k
        x_list.append(np.log10(f(x_k)))
        
        k += 1
    return x_list

def linesearch_armijo(p_k: np.ndarray, x_k: np.ndarray):
    alpha = 1
    rho = 0.5
    c = 10**(-4)
    while f(x_k + alpha*p_k) > f(x_k) + c*alpha*(grad_f(x_k).T @ p_k):
        alpha = alpha*rho
    return alpha

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