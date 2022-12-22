import numpy as np
import matplotlib.pyplot as plt

def dmd(data: np.ndarray, dt, tvals, thrshhld):
    x_0 = data[:,0]
    X = data[:,:-1]
    Y = data[:,1:]

    u, s ,vh = np.linalg.svd(X, full_matrices = False)
    sm = np.max(s)
    indskp = np.log10(s / sm) > -thrshhld
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]
    
    #Solve for Operator
    K = np.conj(ur.T) @ Y @ vr @ np.diag(1./sr)

    #Eigen decomp
    evals, evecs = np.linalg.eig(K)

    #plt.scatter(evals.real, evals.imag)
    #plt.show()

    #modes
    Phi = ur.dot(evecs)
    amps = np.linalg.pinv(Phi) @ x_0

    #reconstruction
    temp = np.repeat(evals[:, None], len(tvals), axis=1)
    tpow = (tvals - tvals[0])/dt
    Psi = np.power(temp, tpow) * amps[:, None]

    recon = Phi.dot(Psi)

    return recon

def cm_dmd(data, NT, thrshhld, tvals, dt):
    # assign values for regression
    x_0 = data[:,0]
    y = data[:,-1]
    X = data[:,:-1]

    #svd and building companion matrix
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    sm = np.max(s)
    indskp = np.log10(s / sm) > -thrshhld
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]
        
    c = vr @ np.diag(1. / sr) @ np.conj(ur.T) @ y
    comp_mat = np.diag(np.array([1.]*(NT-1)), k = -1)
    comp_mat[:,-1] = c

    #calculating eigenvalues/vectors/modes
    evals, evecs = np.linalg.eig(comp_mat)
    modes = X.dot(evecs)
    amps = np.linalg.pinv(modes) @ x_0

    #reconstruction
    Psi = np.vander(evals, N = NT, increasing=True) * amps

    recon = modes.dot(Psi)

    return recon