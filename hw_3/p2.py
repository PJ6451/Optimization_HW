import numpy as np
import conj_grad as cg
import plots as plts

def Hilbert(n):
    i, j = np.ogrid[1:n+1, 1:n+1]
    return 1/(i + j - 1)

n= 20
A = Hilbert(n)

b = np.ones(n)
x_0 = np.zeros(n)

r_list = cg.conj_dir_eig(n, x_0, A, b)
r_array = np.array(r_list)
plts.plot_2(n,r_array)