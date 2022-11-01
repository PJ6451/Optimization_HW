import numpy
import matplotlib.pyplot as plt
import pandas

def f(x):
    ans = 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    return ans

def grad_f(x):
    delt = numpy.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]), 200*(x[1]-x[0]**2)])
    return delt

def hess_f(x):
    delt_2 = numpy.array([[-400*x[1] + 1200*x[0]**2 + 2,-400*x[0]],[-400*x[0],200]])
    return delt_2

def p(x,method):
    if method == 'newton':
        p_k = -1*numpy.dot(numpy.linalg.inv(hess_f(x)),grad_f(x))
    else:
        p_k = grad_f(x)*(-1/numpy.linalg.norm(grad_f(x),2))
    
    return p_k

def backtrack_line_search(x_k,c,rho,tol,alpha,method):
    x_list = [x_k]
    p_list = []
    a_list = [alpha]
    f_list = [f(x_k)]
    while numpy.abs(f(x_k)) > tol:
    #while numpy.linalg.norm(grad_f(x_k),2) > tol:
        p_k = p(x_k,method)
        p_list.append(p_k)
        while f(x_k + alpha*p_k) > f(x_k) + c*alpha*numpy.dot(p_k,grad_f(x_k)):
            alpha = rho*alpha
        
        a_list.append(alpha)
        x_k = x_k + alpha*p_k
        f_list.append(f(x_k))
        x_list.append(x_k)
    
    return x_list, f_list, p_list, a_list

def plot(x,method,part):
    x = numpy.array(x)
    f_k = []
    for x_k in x:
        f_k.append(f(x_k))

    f_k = numpy.array(f_k)
    iters = numpy.arange(1,f_k.size + 1)
    plt.plot(iters, f_k)
    title = method + " with x_0 = (" + str(x[0][0]) + "," + str(x[0][1]) + ")"
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("f(x)")
    filename = method + '_' + part
    plt.savefig(filename)
    plt.clf()

######## ROUND 1 ##########

alpha = 1
rho = 0.5
c = 10**-4
tol = 10**-8
x_k = numpy.array([1.2, 1.2])

x_list, f_list, p_list, a_list = backtrack_line_search(x_k,c,rho,tol,alpha,method = 'newton')

df1 = pandas.DataFrame(x_list)
df2 = pandas.DataFrame(f_list)
df3 = pandas.DataFrame(p_list)
df4 = pandas.DataFrame(a_list)
with pandas.ExcelWriter("path_to_file_1.xlsx") as writer:
    df1.to_excel(writer, sheet_name="Sheet1")
    df2.to_excel(writer, sheet_name="Sheet2")
    df3.to_excel(writer, sheet_name="Sheet3")  
    df4.to_excel(writer, sheet_name="Sheet4") 

alpha = 1
rho = 0.5
c = 10**-4
tol = 10**-8
x_k = numpy.array([1.2, 1.2])

x_list, f_list, p_list, a_list = backtrack_line_search(x_k,c,rho,tol,alpha,method = 'steepest_descent')

df1 = pandas.DataFrame(x_list)
df2 = pandas.DataFrame(f_list)
df3 = pandas.DataFrame(p_list)
df4 = pandas.DataFrame(a_list)
with pandas.ExcelWriter("path_to_file_2.xlsx") as writer:
    df1.to_excel(writer, sheet_name="Sheet1")
    df2.to_excel(writer, sheet_name="Sheet2")
    df3.to_excel(writer, sheet_name="Sheet3")  
    df4.to_excel(writer, sheet_name="Sheet4")  

######## ROUND 2 ##########

alpha = 1
rho = 0.5
c = 10**-4
tol = 10**-8
x_k = numpy.array([-1.2, 1])

x_list, f_list, p_list, a_list = backtrack_line_search(x_k,c,rho,tol,alpha,method = 'newton')

df1 = pandas.DataFrame(x_list)
df2 = pandas.DataFrame(f_list)
df3 = pandas.DataFrame(p_list)
df4 = pandas.DataFrame(a_list)
with pandas.ExcelWriter("path_to_file_3.xlsx") as writer:
    df1.to_excel(writer, sheet_name="Sheet1")
    df2.to_excel(writer, sheet_name="Sheet2")
    df3.to_excel(writer, sheet_name="Sheet3")  
    df4.to_excel(writer, sheet_name="Sheet4") 


alpha = 1
rho = 0.5
c = 10**-4
tol = 10**-8
x_k = numpy.array([-1.2, 1])

x_list, f_list, p_list, a_list = backtrack_line_search(x_k,c,rho,tol,alpha,method = 'steepest_descent')

df1 = pandas.DataFrame(x_list)
df2 = pandas.DataFrame(f_list)
df3 = pandas.DataFrame(p_list)
df4 = pandas.DataFrame(a_list)
with pandas.ExcelWriter("path_to_file_4.xlsx") as writer:
    df1.to_excel(writer, sheet_name="Sheet1")
    df2.to_excel(writer, sheet_name="Sheet2")
    df3.to_excel(writer, sheet_name="Sheet3")  
    df4.to_excel(writer, sheet_name="Sheet4") 
