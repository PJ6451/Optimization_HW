import numpy as np

def spiral(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = -0.2*x1 + x2
    rhs[1] = -x1 - 0.2*x2
    return rhs

def center(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = 4*x1 - 10*x2
    rhs[1] = 2*x1 - 4*x2
    return rhs

def saddle(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x1 + 4*x2
    rhs[1] = 2*x1 - x2
    return rhs

def harmonic(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x2
    rhs[1] = -np.sin(x1)
    return rhs


def duffing(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x2
    rhs[1] = x1 - x1**3.
    return rhs


def lorentz(lhs,sigma,rval,bval):
    y1, y2, y3 = lhs[0], lhs[1], lhs[2]
    rhs = np.zeros(3, dtype=np.float64)
    rhs[0] = sigma*(y2-y1)
    rhs[1] = rval*y1-y2-y1*y3
    rhs[2] = -bval*y3 + y1*y2
    return rhs


# 4th order Runge-Kutta timestepper
def rk4(x0, f, dt):
    k1 = dt*f(x0)
    k2 = dt*f(x0 + k1/2.)
    k3 = dt*f(x0 + k2/2.)
    k4 = dt*f(x0 + k3)
    return x0 + (k1 + 2.*k2 + 2.*k3 + k4)/6.


# Time stepping scheme for solving x' = f(x) for t0<=t<=tf with time step dt.
def timestepper(x0,t0,tf,dt,f):
    ndim = np.size(x0)
    nsteps = np.int((tf-t0)/dt)
    solpath = np.zeros((ndim,nsteps+1),dtype=np.float64)
    solpath[:,0] = x0
    for jj in range(nsteps):
        solpath[:, jj+1] = rk4(solpath[:, jj], f, dt)
    return solpath