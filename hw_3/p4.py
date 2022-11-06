import numpy as np
import conj_grad as cg
import plots as plts
import pandas

def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f(x):
    return np.array([400*x[0]**3 - 400*x[1]*x[0] + 2*x[0] - 2, 200*(x[1]-x[0]**2)])

def hess_f(x):
    return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])

def line_search_newton(x_k: np.ndarray):
    k = 0
    f_list = [np.log10(f(x_k))]
    x_list = [x_k]
    while np.linalg.norm(grad_f(x_k),2) >= 10**(-8):
        nu_k = min(10**(-3), np.sqrt(np.linalg.norm(grad_f(x_k),2)))
        p_k = cg.cg_core(hess_f(x_k), -grad_f(x_k), nu_k)
        alpha_k = linesearch_armijo(p_k, x_k)
        x_k = x_k + alpha_k * p_k
        f_list.append(np.log10(f(x_k)))
        x_list.append(x_k)
        k += 1
    return x_list, f_list

def linesearch_armijo(p_k: np.ndarray, x_k: np.ndarray):
    alpha = 1
    rho = 0.5
    c = 10**(-4)
    while f(x_k + alpha*p_k) > f(x_k) + c*alpha*(grad_f(x_k).T @ p_k):
        alpha = alpha*rho
    return alpha

x_k = np.array([2.8,4])
x_list, f_list = line_search_newton(x_k)
f_array = np.array(f_list)
plts.plot_4(x_k, f_array)

df1 = pandas.DataFrame(x_list)
with pandas.ExcelWriter("path_to_file_1.xlsx") as writer:
    df1.to_excel(writer, sheet_name="p4")