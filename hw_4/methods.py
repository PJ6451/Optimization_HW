import numpy as np
import pandas as pd

def bfgs(x_k: np.ndarray, eps: float, H_k: np.ndarray, f: callable, grad_f: callable, hess_f: callable):
    x_list = [x_k]
    j_list = []
    f_list = [f(x_k)]
    a_list = []
    g_list = [grad_f(x_k)]
    b_list = []
    k = 1
    while np.linalg.norm(grad_f(x_k),2) > eps:
        p_k = -H_k @ grad_f(x_k)
        b_list.append(np.linalg.norm(((H_k - hess_f(x_k)) @ p_k),2)/(np.linalg.norm(p_k,2)))
        alpha_k, j = armijo(p_k, x_k, f, grad_f)
        x_k_1 = x_k + alpha_k * p_k
        s_k = alpha_k * p_k
        y_k = grad_f(x_k_1) - grad_f(x_k)
        y_k = np.reshape(y_k,(2,1))
        s_k = np.reshape(s_k,(2,1))
        rho_k = 1. / (y_k.T @ s_k)
        H_k = (np.eye(2, dtype = float) - (rho_k * (s_k @ y_k.T))) @ H_k @ (np.eye(2, dtype = float) - (rho_k * (y_k @ s_k.T))) + (rho_k * (s_k @ s_k.T))
        x_k = x_k_1

        x_list.append(x_k)
        j_list.append(j)
        f_list.append(f(x_k))
        a_list.append(alpha_k)
        g_list.append(grad_f(x_k))
        k += 1
    
    b_list.append(np.linalg.norm(((H_k - hess_f(x_k)) @ p_k),2)/(np.linalg.norm(p_k,2)))
    z_list = list(zip(x_list, j_list, f_list, a_list, g_list, b_list))
    clmns = ['x_k', 'Inner Iterations', 'f(x_k)', 'alpha','grad_f', 'QN_criteria']
    df1 = pd.DataFrame(z_list, columns=clmns)
    return k, df1, b_list

def armijo(p_k: np.ndarray, x_k: np.ndarray, f: callable, grad_f: callable)-> float:
    j = 1
    alpha = 1.
    rho = 0.5
    c = 10**(-4)
    while f(x_k + alpha*p_k) > f(x_k) + c*alpha*(np.reshape(grad_f(x_k), (2,1))).T @ (np.reshape(p_k, (2,1))):
        alpha = alpha*rho
        j += 1
    return alpha, j

def bfgs_rosen(x_k: np.ndarray, eps: float, H_k: np.ndarray, rosenbrock_2Nd: callable):
    x_list = [x_k]
    j_list = []
    f_list = [rosenbrock_2Nd(x_k,0)]
    a_list = []
    g_list = [rosenbrock_2Nd(x_k,1)]
    b_list = []
    k = 1
    while np.linalg.norm(rosenbrock_2Nd(x_k,1),2) > eps:
        p_k = -H_k @ rosenbrock_2Nd(x_k,1)

        b_list.append(np.linalg.norm(((H_k - rosenbrock_2Nd(x_k,2)) @ p_k),2)/(np.linalg.norm(p_k,2)))
        
        alpha_k, j = armijo_rosen(p_k, x_k, rosenbrock_2Nd)
        x_k_1 = x_k + alpha_k * p_k
        s_k = np.reshape(alpha_k * p_k,(18,1))
        y_k = np.reshape(rosenbrock_2Nd(x_k_1,1) - rosenbrock_2Nd(x_k,1),(18,1))
        
        rho_k = (1. / (y_k.T @ s_k))[0][0]
        LH = np.eye(18) - (rho_k * (s_k @ y_k.T))
        RH = np.eye(18) - (rho_k * (y_k @ s_k.T))
        H_k = LH @ H_k @ RH + (rho_k * (s_k @ s_k.T))
        x_k = x_k_1

        x_list.append(x_k)
        j_list.append(j)
        f_list.append(rosenbrock_2Nd(x_k,0))
        a_list.append(alpha_k)
        g_list.append(rosenbrock_2Nd(x_k,1))
        k += 1
    
    b_list.append(np.linalg.norm(((H_k - rosenbrock_2Nd(x_k,2)) @ p_k),2)/(np.linalg.norm(p_k,2)))
    z_list = list(zip(x_list, j_list, f_list, a_list, g_list, b_list))
    clmns = ['x_k', 'Inner Iterations', 'f(x_k)', 'alpha','grad_f', 'QN_criteria']
    df1 = pd.DataFrame(z_list, columns=clmns)
    return k, df1, b_list

def newton_rosen(x_k: np.ndarray, eps: float, rosenbrock_2Nd: callable):
    x_list = [x_k]
    j_list = []
    f_list = [rosenbrock_2Nd(x_k,0)]
    a_list = []
    g_list = [rosenbrock_2Nd(x_k,1)]
    k = 1
    while np.linalg.norm(rosenbrock_2Nd(x_k,1),2) > eps:
        p_k = -np.linalg.inv(rosenbrock_2Nd(x_k,2)) @ rosenbrock_2Nd(x_k,1)

        alpha_k, j = armijo_rosen(p_k, x_k, rosenbrock_2Nd)
        x_k = x_k + alpha_k * p_k

        x_list.append(x_k)
        j_list.append(j)
        f_list.append(rosenbrock_2Nd(x_k,0))
        a_list.append(alpha_k)
        g_list.append(rosenbrock_2Nd(x_k,1))
        k += 1
    
    z_list = list(zip(x_list, j_list, f_list, a_list, g_list))
    clmns = ['x_k', 'Inner Iterations', 'f(x_k)', 'alpha','grad_f']
    df1 = pd.DataFrame(z_list, columns=clmns)
    return k, df1

def armijo_rosen(p_k: np.ndarray, x_k: np.ndarray, rosenbrock_2Nd: callable)-> float:
    j = 1
    alpha = 1.
    rho = 0.5
    c = 10**(-4)
    while  rosenbrock_2Nd(x_k + alpha*p_k,0) > rosenbrock_2Nd(x_k,0) + c*alpha*(np.reshape(rosenbrock_2Nd(x_k,1), (18,1))).T @ (np.reshape(p_k, (18,1))):
        alpha = alpha*rho
        j += 1
    return alpha, j

def line_search_newton(x_k: np.ndarray, f: callable, grad_f: callable, hess_f: callable):
    k = 1
    x_list = [x_k]
    j_list = []
    f_list = [f(x_k)]
    a_list = []
    g_list = [grad_f(x_k)]
    b_list = []
    while np.linalg.norm(grad_f(x_k),2) >= 10**(-8):
        nu_k = min(10**(-3), np.sqrt(np.linalg.norm(grad_f(x_k),2)))
        p_k, i = cg_core(hess_f(x_k), -grad_f(x_k), nu_k)
        b_list.append(np.linalg.norm(((hess_f(x_k) - hess_f(x_k)) @ p_k),2)/(np.linalg.norm(p_k,2)))
        alpha_k, j = armijo(p_k, x_k, f, grad_f)
        a_list.append(alpha_k)
        x_k = x_k + alpha_k * p_k

        x_list.append(x_k)
        j_list.append(i+j)
        f_list.append(f(x_k))
        a_list.append(alpha_k)
        g_list.append(grad_f(x_k))
        
        k += 1

    b_list.append(np.linalg.norm(((hess_f(x_k) - hess_f(x_k)) @ p_k),2)/(np.linalg.norm(p_k,2)))
    z_list = list(zip(x_list, j_list, f_list, a_list, g_list, b_list))
    clmns = ['x_k', 'Inner Iterations', 'f(x_k)', 'alpha','grad_f', 'QN_criteria']
    df1 = pd.DataFrame(z_list, columns=clmns)
    return k, df1, b_list

def cg_core(A: np.ndarray, b: np.ndarray, nu_k: float):
    j = 1
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
    return x_k, j

def gauss_newton(x_k: np.ndarray, eps: float, r: callable, f: callable, J:callable):
    k = 1
    grad_f = lambda x: (J(x)).T @ r(x)
    while np.linalg.norm(grad_f(x_k),2) > eps:
        p_k = -(np.linalg.pinv(J(x_k)) @ r(x_k))
        alpha_k, _ = armijo(p_k, x_k, f, grad_f)
        x_k = x_k + alpha_k * p_k
        k += 1
    return x_k, k