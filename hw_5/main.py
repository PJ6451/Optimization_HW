import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from pytictoc import TicToc
import warnings

def newtons_method(x_k, r_k, J):
    k = 1
    x_list = [x_k]
    while np.linalg.norm(r_k(x_k),2) > 1e-8:
        if k < 1e5:
            JJ = J(x_k)
            try:
                p_k = -np.linalg.inv(JJ) @ r_k(x_k)
            except np.linalg.LinAlgError:
                k = 'diverge'
                return x_list, k
            x_k = x_k + p_k
            x_list.append(x_k)
            k += 1
        else:
            k = 'diverge'
            return x_list, k           
    return x_list, k

def steepest_descent_bt(x_k, r_k, J):
    k = 1
    rk = r_k(x_k)
    f = lambda x: 0.5*np.dot(r_k(x), r_k(x))
    grad_f = lambda x: J(x).T.dot(r_k(x))
    while np.linalg.norm(r_k(x_k),2) > 1e-8:
        if k < 1e5:
            p_k = -(J(x_k).T).dot(rk)
            alpha = armijo(p_k, x_k, f, grad_f)
            x_k = x_k + alpha*p_k
            rk = r_k(x_k)
            k += 1
        else:
            k = 'diverge'
            x_k = 'diverge'
            return x_k, k
    return x_k, k

def armijo(p_k: np.ndarray, x_k: np.ndarray, f: callable, grad_f: callable)-> float:
    alpha = 1.
    rho = 0.5
    c = 10**(-4)
    while f(x_k + alpha*p_k) > f(x_k) + c*alpha*(np.reshape(grad_f(x_k), (3,1))).T @ (np.reshape(p_k, (3,1))):
        alpha = alpha*rho
    return alpha

def steepest_descent_e(x_k, r_k, J):
    k = 1
    while np.linalg.norm(r_k(x_k),2) > 1e-8:
        if k < 1e5:
            p_k = -(J(x_k).T).dot(r_k(x_k))
            v = (J(x_k) @ J(x_k).T).dot(r_k(x_k))
            alpha = (np.dot(v,r_k(x_k))) / (np.dot(v,v))
            x_k = x_k + alpha*p_k
            k += 1
        else:
            k = 'diverge'
            x_k = 'diverge'
            return x_k, k
    return x_k, k

def get_ic():
    ic = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,1,1]
    ])
    return ic

def p1():
    r_k = lambda x: np.array([
        x[0]+x[1]-3, 
        x[0]**2 + x[1]**2 - 9
    ], dtype=float)
    J = lambda x: np.array([
        [1., 1.],
        [2*x[0], 2*x[1]]
    ], dtype=float)
    x_k = np.array([1,5])
    x_list, _ = newtons_method(x_k, r_k, J)
    df = pd.DataFrame(x_list, columns=['x_k_1', 'x_k_2'])
    with pd.ExcelWriter("p1.xlsx") as writer:
        df.to_excel(writer)

def p2():
    ic = get_ic()
    r_k = lambda x: np.array([
        x[0]**2 + x[1]**2 + x[2]**2 - 1,
        x[0] + x[1] + x[2],
        x[0] - x[1]**2
    ], dtype=float)
    J = lambda x: np.array([
        [2*x[0], 2*x[1], 2*x[2]],
        [1., 1., 1.],
        [1., -2*x[1], 0]
    ], dtype=float)
    i_list_1 = []
    t_list_1 = []
    x_list_1 = []
    i_list_2 = []
    t_list_2 = []
    t = TicToc()
    for i in ic:
        #newton
        t0 = t.tic()
        x_list, k = newtons_method(i, r_k, J)
        tf = t.tocvalue()*1000
        if k == 'diverge':
            tf = 'diverge'
        i_list_1.append(k)
        t_list_1.append(tf)
        x_list_1.append(i)
        #steepest descent
        t0 = t.tic()
        x_k, k = steepest_descent_bt(i, r_k, J)
        tf = t.tocvalue()*1000
        if k == 'diverge':
            tf = 'diverge'
        i_list_2.append(k)
        t_list_2.append(tf)

    z_list = list(zip(x_list_1, i_list_1, t_list_1, i_list_2, t_list_2))
    clmns = [
        'guess',
        'Number of Iterations 1', 
        'Computing Time 1',
        'Number of Iterations 2', 
        'Computing Time 2'
        ]
    df = pd.DataFrame(z_list, columns=clmns)
    with pd.ExcelWriter("p2.xlsx") as writer:
        df.to_excel(writer)

def p3():
    ic = get_ic()
    r_k = lambda x: np.array(
        [x[0]**2 + x[1]**2 + x[2]**2 - 1,
        x[0] + x[1] + x[2],
        x[0] - x[1]**2]
    , dtype=float)
    J = lambda x: np.array([
        [2*x[0], 2*x[1], 2*x[2]],
        [1, 1, 1],
        [1, -2*x[1], 0]
    ], dtype=float)
    i_list = []
    t_list = []
    x_list = []
    for i in ic:
        t = TicToc()
        t0 = t.tic()
        x_k, k = steepest_descent_e(i, r_k, J)
        tf = t.tocvalue()*1000
        if k == 'diverge':
            tf = 'diverge'
        i_list.append(k)
        t_list.append(tf)
        x_list.append(i)
    z_list = list(zip(x_list, i_list, t_list))
    clmns = [
        'x_k',
        'Number of Iterations', 
        'Computing Time'
        ]
    df = pd.DataFrame(z_list, columns=clmns)
    with pd.ExcelWriter("p3.xlsx") as writer:
        df.to_excel(writer)

if __name__ == '__main__':
    #p1()
    p2()
    #p3()