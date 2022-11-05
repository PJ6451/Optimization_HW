import numpy as np

def standard_cg(x_k: np.ndarray, A: np.ndarray, b: np.ndarray):
    k = 0
    r_k = np.dot(A, x_k) - b
    r_list = [np.log10(np.linalg.norm(r_k,2))]
    p_k = -r_k
    while np.linalg.norm(r_k,2) > 10**(-6):
        rTr = np.dot(r_k, r_k)
        Apk = np.dot(A, p_k)
        alpha_k =  (rTr) / (np.dot(p_k, Apk))
        x_k = x_k + alpha_k * p_k
        r_k = r_k + alpha_k * Apk
        beta_k = (np.dot(r_k, r_k)) / (rTr)
        p_k = -r_k + beta_k * p_k
        r_list.append(np.log10(np.linalg.norm(r_k,2)))
        k += 1
    print(k)
    return r_list

def conj_dir_eig(n: int, x_k: np.ndarray, A: np.ndarray, b: np.ndarray):
    r_list = []
    _, evecs = np.linalg.eig(A)
    for k in range(n):
        r_k = A @ x_k - b
        r_list.append(np.log10(np.linalg.norm(r_k,2)))
        p_k = evecs[:,k]
        alpha_k =  -(r_k.T @ p_k) / (p_k.T @ A @ p_k)
        x_k = x_k + alpha_k * p_k
        r_k = A @ x_k - b
    return r_list

def standard_cg_get_min(x_k: np.ndarray, A: np.ndarray, b: np.ndarray):
    r_k = A @ x_k - b
    p_k = -r_k
    while np.linalg.norm(r_k,2) > 10**(-6):
        rTr = r_k.T @ r_k
        Apk = A @ p_k
        alpha_k =  (rTr) / (p_k.T @ Apk)
        x_k = x_k + alpha_k * p_k
        r_k_1 = r_k + alpha_k * Apk
        beta_k = (r_k_1.T @ r_k_1) / (rTr)
        p_k = -r_k_1 + beta_k * p_k
        r_k = r_k_1
    return x_k

def standard_cg_op_norm(x_star: np.ndarray, x_k: np.ndarray, A: np.ndarray, b: np.ndarray):
    vec = x_k - x_star
    x_list = [np.log(vec.T @ A @ vec)]
    r_k = A @ x_k - b
    p_k = -r_k
    while np.linalg.norm(r_k,2) > 10**(-6):
        rTr = r_k.T @ r_k
        Apk = A @ p_k
        alpha_k =  (rTr) / (p_k.T @ Apk)
        x_k = x_k + alpha_k * p_k
        r_k_1 = r_k + alpha_k * Apk
        beta_k = (r_k_1.T @ r_k_1) / (rTr)
        p_k = -r_k_1 + beta_k * p_k
        r_k = r_k_1

        vec = x_k - x_star
        x_list.append(np.log(vec.T @ A @ vec))
    return x_list