import numpy as np
import conj_grad as cg
import plots as plts

n = 20
b = np.ones(n)
x_0 = np.zeros(n)
 
k = np.append(
    np.array([95, 100, 115, 150, 125]),
    np.arange(.95,1.05,((1.05-.95)/15)),
    0
)

A = np.diag(k)
x_star = cg.standard_cg_get_min(x_0, A, b)
r_list_1 = cg.standard_cg_op_norm(x_star, x_0, A ,b)
r_array_1 = np.array(r_list_1)

k = np.random.uniform(0, 100, n)
A = np.diag(k)
x_star = cg.standard_cg_get_min(x_0, A, b)
r_list_2 = cg.standard_cg_op_norm(x_star, x_0, A ,b)
r_array_2 = np.array(r_list_2)

plts.plot_3(r_array_1, r_array_2)