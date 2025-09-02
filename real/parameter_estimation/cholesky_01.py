import numpy as np
import scipy
from scipy.integrate import solve_ivp

def cholesky(df, theta, t_eval, model_func):
    h = 1e-6

    def solve(params, t_eval):
        z0 = [params[4], params[5]]  # Initial conditions from parameter vector
        def ode(t, z):
            return model_func(t, z, params[:4])  # Only pass first 4 params to model_func
        t_span = (t_eval[0], t_eval[-1])
        sol = solve_ivp(ode, t_span, z0, t_eval=t_eval, method='RK45')
        return sol

    F = solve(theta, t_eval).y.T
    N, d = F.shape
    p = len(theta)

    def error_matrix():
        d_obs = df[['prey', 'pred']].values
            # Compute errors in log-space for lognormal noise
        log_d_obs = np.log(d_obs + 1e-8)
        log_F = np.log(F + 1e-8)
        diag_sum = np.zeros((2,2))
        for i in range(N):
            diff = log_d_obs[i] - log_F[i]
            outer = np.outer(diff, diff)
            diag_sum += np.diag(np.diag(outer))
        Sigma = diag_sum / (N-p)
        Sigma_diag = np.diag(Sigma)
        Sigma_diag_neg_half = np.power(Sigma_diag, -0.5)
        Sigma_root = np.diag(Sigma_diag_neg_half)
        return Sigma_root

    def sensitivity_matrix():
        S = np.zeros((N, d, p))
        for k in range(p):
            theta_shift = theta.copy()
            theta_shift[k] += h
            F_shift = solve(theta_shift, t_eval).y.T
            S[:, :, k] = (F_shift - F) / h
        return S

    def X_matrix():
        X = np.zeros((N * d, p))
        S = sensitivity_matrix()
        Sigma = error_matrix()
        for i in range(N):
            block = Sigma @ S[i]
            X[i*d: (i+1)*d, :] = block
        return X

    def qr_decomp():
        X = X_matrix()
        Q, R = np.linalg.qr(X)
        return R

    def invert_R():
        R = qr_decomp()
        I = np.eye(R.shape[0])
        R_inv = scipy.linalg.solve_triangular(R, I, lower=False)
        return R_inv

    R = invert_R()
    return R