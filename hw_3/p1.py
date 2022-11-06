import numpy as np
import conj_grad as cg
import plots as plts

def Hilbert(n):
    i, j = np.ogrid[1:n+1, 1:n+1]
    return 1/(i + j - 1)

n_list = [5,8,12,20]
iter_list = []
cond_list = []
eig_5_list = []
eig_8_list = []
eig_12_list = []
eig_20_list = []

for n in n_list:
    A = Hilbert(n)
    
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
    aaa = np.log10((np.linalg.cond(A)-1)/(np.linalg.cond(A)+1))
    bbb = np.log10((np.sqrt(np.linalg.cond(A))-1)/(np.sqrt(np.linalg.cond(A))+1))

    b = np.ones(n)
    x_0 = np.zeros(n)

    r_list = cg.standard_cg(x_0, A, b)
    r_array = np.array(r_list)
    iter_list.append(r_array.size-1)
    print((bbb/aaa)*(r_array.size-1))
    plts.plot_1_a(n,r_array)

plts.plot_1_b(n_list, iter_list)
plts.plot_1_c(n_list, cond_list)
plts.plot_1_d(eig_5_list, eig_8_list, eig_12_list, eig_20_list)