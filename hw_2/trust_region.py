import numpy
import matplotlib.pyplot as plt
import pandas
from matplotlib import ticker

def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f(x):
    return numpy.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]), 200*(x[1]-x[0]**2)])

def hess_f(x):
    return numpy.array([[-400*x[1] + 1200*x[0]**2 + 2,-400*x[0]],[-400*x[0],200]])

def m_k(p,B_k,grad,fv):
    return fv + numpy.dot(grad,p) + 0.5 * numpy.dot(p,numpy.dot(B_k,p)) 

def p(method,del_k,B_k,grad):
    gt_B_g = numpy.dot(numpy.matmul(grad,B_k),grad)
    if method == 'cauchy_point':
        nrm = numpy.linalg.norm(grad,2)
        if gt_B_g <= 0:
            tau_k = 1
        else:
            tau_k = min([1,(nrm**3)/(del_k*gt_B_g)])
        return -tau_k * del_k/nrm * grad
    elif method == 'dogleg':
        g2 = numpy.dot(grad, grad)
        p_u_k = -(g2/gt_B_g)*grad
        p_fs_k = -numpy.dot(numpy.linalg.inv(B_k), grad)
        p_u_norm = numpy.linalg.norm(p_u_k,2)
        tau = del_k/p_u_norm
        if tau <= 1:
            return tau*p_u_k
        elif tau <= 2:
            return p_u_k + (tau-1)*(p_fs_k - p_u_k)
        else:
            p_fs_norm = numpy.linalg.norm(p_fs_k,2)
            if p_fs_norm <= del_k:
                return p_fs_k
            else:
                return (del_k/p_fs_norm)*p_fs_k

def trust_region(x_k,nu,del_k,method,del_hat):
    k = 0
    x_list = [x_k]
    while numpy.linalg.norm(grad_f(x_k),2) > 10**(-8):
        ###### Assign Values #####
        B_k = hess_f(x_k)
        grad = grad_f(x_k)
        fv = f(x_k)
        k += 1
        
        ###### Get p_k #####
        p_k = p(method,del_k,B_k,grad)

        ##### Calculate rho_k #####
        num = f(x_k) - f(x_k + p_k)
        den = m_k(numpy.array([0,0]),B_k,grad,fv) - m_k(p_k,B_k,grad,fv)
        rho_k = num/den

        ##### Calculate new del_k, x_k ######
        if rho_k < 1/4:
            del_k = (1/4)*del_k
        else:
            if rho_k > 3/4:
                if numpy.linalg.norm(p_k,2) == del_k:
                    del_k = min([2*del_k,del_hat])
        if rho_k > nu:
            x_k = x_k + p_k
        x_list.append(x_k)
    
    return x_list

def plot(x,method,num):
    x_1 = numpy.array(x)

    X, Y = numpy.meshgrid(numpy.linspace(-num, num, 100), numpy.linspace(-num, num, 100))
    z = 100*(Y - X**2)**2 + (1 - X)**2

    fig, ax = plt.subplots(1, 1)
    cf = ax.contourf(X, Y, z, locator = ticker.LogLocator(), cmap = 'bone')
    fig.colorbar(cf, ax=ax)
    plt.plot(x_1[:,0],x_1[:,1],marker='o',markersize=5, color ='r')
    title = method + " with x_0 = (" + str(x_1[0][0]) + "," + str(x_1[0][1]) + ")"
    ax.set_title(title)
    if x_1[0][0] == -1.2:
        part = 'x_0'
    else:
        part = 'x_1'
    filename = method + '_' + part
    plt.savefig(filename)
    plt.clf()

def plot_iters(x,method):
    x = numpy.array(x)
    f_k = []
    for x_k in x:
        f_k.append(f(x_k))

    f_k = numpy.array(f_k)
    iters = numpy.arange(1,f_k.size + 1)
    plt.semilogy(iters, f_k)
    title = method + " with x_0 = (" + str(x[0][0]) + "," + str(x[0][1]) + ")"
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("f(x)")
    if x[0][0] == -1.2:
        part = 'x_0'
    else:
        part = 'x_1'

    filename = method + '_iters_' + part
    plt.savefig(filename)
    plt.clf()

del_k = 1
del_hat = 300
nu = 1/4
x_k = numpy.array([-1.2,1])
method = 'dogleg'

x_list_1 = trust_region(x_k,nu,del_k,method,del_hat)
plot(x_list_1,method,5)
plot_iters(x_list_1,method)

x_k = numpy.array([2.8,4])
method = 'dogleg'

x_list_2 = trust_region(x_k,nu,del_k,method,del_hat)
plot(x_list_2,method,8)
plot_iters(x_list_2,method)

x_k = numpy.array([-1.2,1])
method = 'cauchy_point'

x_list_3 = trust_region(x_k,nu,del_k,method,del_hat)
plot(x_list_3,method,5)
plot_iters(x_list_3,method)

x_k = numpy.array([2.8,4])
method = 'cauchy_point'

x_list_4 = trust_region(x_k,nu,del_k,method,del_hat)
plot(x_list_4,method,5)
plot_iters(x_list_4,method)

df1 = pandas.DataFrame(x_list_1)
df2 = pandas.DataFrame(x_list_2)
df3 = pandas.DataFrame(x_list_3)
df4 = pandas.DataFrame(x_list_4)
with pandas.ExcelWriter("path_to_file_1.xlsx") as writer:
    df1.to_excel(writer, sheet_name="dogleg_x_1")
    df2.to_excel(writer, sheet_name="dogleg_x_2")
    df3.to_excel(writer, sheet_name="cauchy_point_x_1")  
    df4.to_excel(writer, sheet_name="cauchy_point_x_2") 