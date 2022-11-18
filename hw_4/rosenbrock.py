import numpy as np

def f(x):
    num = 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    return num

def grad_f(x):
    vec = np.array([400*x[0]**3 - 400*x[1]*x[0] + 2*x[0] - 2, 200*(x[1]-x[0]**2)])
    return vec

def hess_f(x):
    mat =  np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])
    return mat

def rosenbrock_2Nd(x,order):
    if order == -1:
        xN     = np.array([  1 , 1 ])
        x0easy = np.array([  1.2 , 1.2 ])
        x0e2   = (xN + x0easy) / 2
        x0e3   = (xN + x0e2)  / 2
        x0e4   = (xN + x0e3) / 2
        x0hard = np.array([ -1.2 , 1.0 ])
        x0h2   = (xN + x0hard) / 2
        x0h3   = (xN + x0h2)  / 2
        x0h4   = (xN + x0h3) / 2
        x0h5   = 2*x0hard
        R      = np.array([x0easy, x0e2, x0e3, x0e4, x0hard, x0h2, x0h3, x0h4, x0h5])
        return np.reshape(R,[18])
    
    nx = 18

    rb2d      = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    rb2d_x    = lambda x: -400*(x[1]-x[0]**2)*x[0]-2+2*x[0]
    rb2d_xx   = lambda x: 1200*x[0]**2-400*x[1]+2 
    rb2d_xy   = lambda x: -400*x[0]
    rb2d_y    = lambda x: 200*x[1]-200*x[0]**2
    rb2d_yy   = lambda x: 200
    rb2d_grad = lambda x: np.array( [ rb2d_x(x) , rb2d_y(x) ] )
    rb2d_hess = lambda x: np.array([ [ rb2d_xx(x) , rb2d_xy(x) ],[ rb2d_xy(x) , rb2d_yy(x)] ])

    if order == 0:
        R = 0
        for k in range(0, nx, 2):
            R += rb2d(x[k:(k+2)])
    elif order == 1:
        R = np.zeros(len(x))
        for k in range(0, nx, 2):
            R[k:(k+2)] = rb2d_grad(x[k:(k+2)])
    elif order == 2:
        R = np.zeros([len(x),len(x)])
        for k in range(0, nx, 2):
            R[k:(k+2),k:(k+2)] = rb2d_hess(x[k:(k+2)])
    
    return R