import numpy as np
import methods
import rosenbrock as rs
import matplotlib.pyplot as plt
import pandas as pd

def p1():
    H = np.eye(18, dtype = float)
    eps = 10**(-5)
    x_0 = rs.rosenbrock_2Nd(1,-1)

    k, df1, b_list = methods.bfgs_rosen(x_0, eps, H, rs.rosenbrock_2Nd)
    _, df2 = methods.newton_rosen(x_0, eps, rs.rosenbrock_2Nd)
    plt.plot(np.arange(k),np.log10(np.array(b_list)))
    plt.title('Quasi-Newton Condition for BFGS method')
    plt.xlabel('Iteration $k$')
    plt.ylabel("$log_{10}$")
    plt.show()
    with pd.ExcelWriter("p1.xlsx") as writer_1:
        df1.to_excel(writer_1, sheet_name='bfgs')
        df2.to_excel(writer_1, sheet_name='newton')


def p2():
    x_k = np.array([0,0])
    f = lambda x: 0.5*5*4*(-0.02*x[0] + 0.5*x[0]**2 + x[1])**2 + 0.5*(-0.02 + 0.5*x[0]**2 - (1./4)*x[1])*4**4 - 4*x[0]*2e-5
    grad_f = lambda x: np.array([5*4*(-0.02*x[0]+0.5*x[0]**2 + x[1])*(x[0]-0.02) + (0.5)*(4**4)*(x[0]-0.02) - 2e-5*4, 5*4*(-0.02*x[0]+0.5*x[0]**2 + x[1]) - 0.5*4**3])
    H = np.eye(2)
    eps = 10**(-5)
    _, df1, _ = methods.bfgs(np.array(x_k), eps, H, f, grad_f, rs.hess_f)
    with pd.ExcelWriter("p2_bfgs.xlsx") as writer_1:
        df1.to_excel(writer_1)

def p4():
    x_k = np.array([0,0])
    eps = 1e-7
    r = lambda x: np.array([np.exp(x[0] - 2.*x[1])-5, np.exp(x[0] - x[1])-1, np.exp(x[0])-2, np.exp(x[0] + x[1])+4])
    f = lambda x: 0.5* (r(x).T @ r(x))
    J = lambda x: np.array([
        [np.exp(x[0]-2*x[1]), -2*np.exp(x[0]-2*x[1])],
        [np.exp(x[0]-x[1]), -np.exp(x[0]-x[1])],
        [np.exp(x[0]), 0],
        [np.exp(x[0]+x[1]), np.exp(x[0]+x[1])]
        ])
    x_min, _ = methods.gauss_newton(x_k, eps, r, f, J)
    print(x_min)
    t = [-2, -1, 0, 1]
    y = [5, 1, 2, -4]
    g = np.zeros(4)
    for i in range(4):
        g[i] = np.exp(x_min[0] + t[i]*x_min[1])
    plt.scatter(t,y, label='$(t_i, y_i)$')
    plt.plot(t, g, label='$g(t)$')
    plt.title('Problem 4.ii')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    p1()
    p2()
    p4()