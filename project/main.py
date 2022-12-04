import numpy as np
import matplotlib.pyplot as plt
from numerical_solvers import *
from dmd import *

def make_plots(rawdata, recon, tvals, numplots, lbl):
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(1, 2, 1)
    for i in range(numplots):
        if i == 1:
            ax.plot(rawdata[i,0,:], rawdata[i,1,:],label='RK4')
            ax.scatter(recon[i,0,:], recon[i,1,:],label='DMD')
        else:
            ax.plot(rawdata[i,0,:], rawdata[i,1,:])
            ax.scatter(recon[i,0,:], recon[i,1,:])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    title = 'Phase Plane for ' + lbl + ' System'
    plt.title(title)
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    error = np.zeros([len(tvals)])
    for i in range(len(tvals)):
        error[i] = np.linalg.norm(rawdata[:,:,i]-recon[:,:,i], 2)
    ax.plot(tvals,np.log10(error))
    plt.xlabel('Time')
    ax.set_ylabel('$\log_{10} || r_k ||_{2}$',rotation=-90,labelpad=15)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.title('Error of reconstruction over time')

    fig.savefig("dmd_project_" + lbl, dpi=200)

def raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt):
    #stack data
    stacked_data = np.zeros([numiconds*numdim, NT+1])
    for i in range(numdim):
        stacked_data[(i)*numiconds:(i+1)*numiconds,:] = rawdata[:,i,:]

    #dmd
    thrshhld = 15
    recon = dmd(stacked_data, dt, tvals, thrshhld)

    #unstack data
    unstacked_data = np.zeros([numiconds, numdim, NT+1])
    for i in range(numdim):
        unstacked_data[:,i,:] = np.real(recon[(i)*numiconds:(i+1)*numiconds,:])

    return unstacked_data

def cent():
    dt = .05
    t0 = 0.
    tf = 10
    NT = int((tf-t0)/dt)
    tvals = np.linspace(t0,tf,NT+1)
    numiconds = 4
    initconds = np.array([
        [1,2],
        [-4,3],
        [1.5,-3],
        [-1.2,-3.6]
    ])
    numdim = 2
    rawdata = np.zeros([numiconds, numdim, NT+1], dtype=np.float64)
    fhandle = lambda x: center(x)
    for ll in range(numiconds):
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #dmd
    recon = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #plot
    make_plots(rawdata, recon, tvals, 4, 'Center')

def sadd():
    dt = .05
    t0 = 0.
    tf = .5
    NT = int((tf-t0)/dt)
    tvals = np.linspace(t0,tf,NT+1)
    numiconds = 4
    initconds = np.array([
        [17,12],
        [-14,13],
        [10,-13],
        [-10.2,-13.6]
    ])
    numdim = 2
    rawdata = np.zeros([numiconds, numdim, NT+1], dtype=np.float64)
    fhandle = lambda x: saddle(x)
    for ll in range(numiconds):
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #dmd
    recon = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #plot
    make_plots(rawdata, recon, tvals, 4, 'Saddle')

def spir():
    dt = .05
    t0 = 0.
    tf = 10
    NT = int((tf-t0)/dt)
    tvals = np.linspace(t0,tf,NT+1)
    numiconds = 4
    initconds = np.array([
        [1,2],
        [-4,3],
        [1.5,-3],
        [-1.2,-3.6]
    ])
    numdim = 2
    rawdata = np.zeros([numiconds, numdim, NT+1], dtype=np.float64)
    fhandle = lambda x: spiral(x)
    for ll in range(numiconds):
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #dmd
    recon = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #plot
    make_plots(rawdata, recon, tvals, 4, 'Spiral Attractor')

def harm():
    dt = .05
    t0 = 0.
    tf = 10
    NT = int((tf-t0)/dt)
    tvals = np.linspace(t0,tf,NT+1)
    numiconds = 80
    numdim = 2
    initconds = np.zeros((numiconds,numdim), dtype=np.float64)
    rawdata = np.zeros([numiconds, numdim, NT+1], dtype=np.float64)
    fhandle = lambda x: harmonic(x)
    for ll in range(numiconds):
        initconds[ll,:] = np.random.uniform(-5.,5.,numdim)
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #dmd
    recon = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #plot
    make_plots(rawdata, recon, tvals, 4, 'Harmonic Oscillator')

def duff():
    dt = .05
    t0 = 0.
    tf = 10.
    NT = int((tf-t0)/dt)
    tvals = np.linspace(t0,tf,NT+1)
    numdim = 2
    numiconds = 80
    initconds = np.zeros((numiconds,2), dtype=np.float64)
    rawdata = np.zeros([numiconds, numdim, NT+1], dtype=np.float64)
    fhandle = lambda x: duffing(x)
    for ll in range(numiconds):
        initconds[ll,:] = np.random.uniform(-3.,3.,numdim)
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #dmd
    recon = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #plot
    make_plots(rawdata, recon, tvals, 4, 'Duffing Oscillator')

def lorenz():
    sigma = 10.
    bval = 8./3.
    rval = 28.
    dt = .05
    t0 = 0.
    tf = 90.
    NT = int((tf-t0)/dt)
    tvals = np.linspace(t0,tf,NT+1)
    numiconds = 80
    numdim = 3
    initconds = np.zeros((numiconds,3), dtype=np.float64)
    rawdata = np.zeros([numiconds, numdim, NT+1], dtype=np.float64)
    x0 = np.array([3.1, 3.1, rval-1])
    fhandle = lambda x: lorentz(x,sigma,rval,bval)
    for ll in range(numiconds):
        initconds[ll,:] = 8.*(np.random.rand(numdim) - .5) + x0
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)
    
    #dmd
    recon = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    fig = plt.figure(figsize = (10, 7))
    traj = rawdata[0,:,:]
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot3D(traj[0,:], traj[1,:], traj[2,:],label='rk4')
    ax.legend()
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot3D(recon[0,0,:], recon[0,1,:], recon[0,2,:],label='DMD')
    ax.legend()
    fig.savefig("dmd_project_lorenz", dpi=200)

    fig = plt.figure(figsize = (10, 7))
    error = np.zeros([len(tvals)])
    for i in range(len(tvals)):
        error[i] = np.linalg.norm(rawdata[:,:,i]-recon[:,:,i], 2)
    plt.plot(tvals,np.log10(error))
    plt.xlabel('Time')
    plt.title('Error of reconstruction over time')
    fig.savefig("dmd_project_lorenz_error", dpi=200)

if __name__ == '__main__':
    #cent()
    #sadd()
    #spir()
    #harm()
    #duff()
    lorenz()