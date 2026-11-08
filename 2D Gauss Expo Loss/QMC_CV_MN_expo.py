import numpy as np
import time
import qmcpy
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from scipy.linalg import solve_triangular

from functools import partial
from collections import defaultdict

import logging

logging.basicConfig(level=logging.INFO)

from msra_loss import CommonLossFunctionAbs


# ============================================================================
# RQMC estimators (Gaussian IS with domain transformation)
# Notes:
# - `m_prev` is included to evaluate difference-integrands across iterations.
# ============================================================================
def RQMC_Fourier_1D_MN_vec(K, idx, m_val, m_prev, N, m, sigma_IS, integrand_func, qmc_seq=None):
    """Compute the 1D RQMC Fourier estimator and per-shift means."""
    sigma_IS = np.atleast_1d(sigma_IS).item()
    log_norm_const = -0.5*np.log(2*np.pi) - np.log(sigma_IS)

    # (m, N) Sobol uniforms
    if qmc_seq is None:
        xi = np.empty((m, N), dtype=float)
        for s in range(m):
            xi[s] = qmcpy.DigitalNetB2(1, graycode=True, randomize="DS", seed=s).gen_samples(N).ravel()
    else:
        xi = np.asarray(qmc_seq, dtype=float).reshape(m, N)

    U = norm.ppf(xi) * sigma_IS         

    F = integrand_func(U, idx, m_val, K,m_prev = m_prev)

    quad = (U / sigma_IS)**2              
    log_IS = log_norm_const - 0.5*quad
    W = np.exp(-log_IS)                   

    V = np.real(F) * W                   
    V_shift_means = V.mean(axis=1)       
    return V_shift_means.mean(), V_shift_means

# ---------------------------------------------------------------------------
# 2D estimator
# ---------------------------------------------------------------------------
def RQMC_Fourier_2D_MN_vec(K, idx1, idx2, m_val, m_prev, N, m, sigma_IS,L_IS, integrand_func, qmc_seq_2D=None):
    """Compute the 2D RQMC Fourier estimator and per-shift means."""
    m1,m2 = m_val
    
    L = np.asarray(L_IS)
    log_norm_const = -np.log(2*np.pi) - 0.5*np.log(np.linalg.det(sigma_IS))

    if qmc_seq_2D is None:
        xi = np.empty((m, N, 2), dtype=float)
        for s in range(m):
            xi[s] = qmcpy.DigitalNetB2(2, graycode=True, randomize="DS", seed=s).gen_samples(N)
    else:
        xi = np.asarray(qmc_seq_2D, dtype=float).reshape(m, N, 2)

    Z = norm.ppf(xi)             
    U = Z @ L.T                 

    
    F = integrand_func(U[...,0], U[...,1], idx1, idx2, m1, m2, K,xy_prev= m_prev)   

    
    U2 = U.reshape(-1, 2)
    W = solve_triangular(L, U2.T, lower=True, overwrite_b=False).T  
    quad = (W**2).sum(axis=1).reshape(m, N)

    log_IS = log_norm_const - 0.5*quad
    V = np.real(F) * np.exp(-log_IS)     
    V_shift_means = V.mean(axis=1)
    return V_shift_means.mean(), V_shift_means


# ============================================================================
# Multilevel Fourier-RQMC class
# ============================================================================
class RQMC_CV_Fou_MN_expo(CommonLossFunctionAbs):
    def __init__(
        self,
        v_mu,
        v_sigma,
        v_beta,
        N_sobol=64,
        m_shift=30,
        alpha=0,
        c=None,
        epsilon=1,
    ):
        """
        Initialize the Multilevel Fourier-RQMC solver for exponential loss.

        Parameters:
        - v_mu: Mean vector.
        - v_sigma: Covariance matrix (or a vector reshaped into a matrix).
        - v_beta: Risk-aversion coefficient.
        - N_sobol: Number of Sobol points per shift (default: 64).
        - m_shift: Number of randomized digital shifts (default: 30).
        - alpha: System-level coupling weight.
        - c: Optional inequality-constraint constant.
        - epsilon: Scale used in domain transformation.
        """
        self._v_mu = np.array(v_mu)
        self._v_sigma = np.array(v_sigma).reshape((len(v_mu), len(v_mu)))
        self._v_beta = v_beta
        self._alpha = alpha
        self._epsilon = epsilon

        self._1_alpha = 1 / (1 + self._alpha)

        self._N_current = int(N_sobol)
        self._N_fixed = int(N_sobol)
        self._m_shift = int(m_shift)
        

        d = len(self._v_mu)

        
        # Common constants for Fourier inversion.
        self._coeff_1D = 0.5 / np.pi

        self._coeff_2D = 1 / (2 * np.pi) ** 2

        

        self._current_m = None 

        # ---------------------------------------------------------------------------
        # Last damping parameters by component family.
        # For 1D terms (`g`).
        # ---------------------------------------------------------------------------
        self._last_K1 = defaultdict(lambda: None)
        self._history_1d = defaultdict(
            lambda: {"K": []}
        )
        self._cache_1d_val = {}
        self._cache_1d_V_list = {}

        # For 2D terms (`h`).
        self._last_K2 = defaultdict(lambda: None) 
       
        self._history_2d = defaultdict(
            lambda: {"K": []}
        )
        # key: (i, j, sx, sy)

        self._cache_2d_val = (
            {}
        ) 
        self._cache_2d_V_list = {}

        # Sobol caches and randomization state.
        self._sobol_1d_cache = {}
        self._sobol_2d_cache = {}

        # ---------------------------------------------------------------------------
        # Sobol backbone and shift bookkeeping for digital-shift randomization.
        # ---------------------------------------------------------------------------
        self._DIGITAL_SHIFT_BITS = 52
        self._DIGITAL_SHIFT_SCALE = 1.0 / (1 << self._DIGITAL_SHIFT_BITS)
        self._sobol_base = {}
        self._sobol_base_bits = {}
        self._sobol_seed_map_1d = {}
        self._sobol_seed_map_2d = {}
        self._sobol_seed_1d = 0
        self._sobol_seed_2d = 0
        
        # ---------------------------------------------------------------------------
        # Transform covariance for domain transformation.
        # ---------------------------------------------------------------------------
        self._sigma_trans_1D = [self._rescale_cov(self._v_sigma[i,i]) for i in range(d)]
        self._sigma_trans_2D = {}
        self._L_2D = {}
        swap = np.array([1, 0])

        for i in range(d):
            for j in range(i + 1, d):
                # build canonical block once
                mat = self._rescale_cov(self._v_sigma[np.ix_([i, j], [i, j])])
                chol =  np.linalg.cholesky(mat)

                # store canonical key
                self._sigma_trans_2D[(i, j)] = mat
                self._L_2D[(i, j)] = chol

                # Factor the reversed covariance to preserve lower triangularity.
                self._sigma_trans_2D[(j, i)] = mat[np.ix_(swap, swap)]
                self._L_2D[(j, i)] = np.linalg.cholesky(
                    self._sigma_trans_2D[(j, i)]
                )
        
        # ---------------------------------------------------------------------------
        # Set up for difference integrands.
        # ---------------------------------------------------------------------------
        
        # State used for difference-integrand accumulation across iterations.
        self._prev_m_loss       = None
        self._prev_m_jac = None
        self._cv_history = []
        self._pending_cv_step = None
        self._prev_m_cv = None
        
        self._accum_loss      = 0.0
        self._accum_jac       = None
        self._jac_components = []          
        self._pending_jac = None
        
        #  integration counts.

        # Memoized objective and gradient at the last queried point.
        self._last_loss_m = None
        self._last_loss_val = None
        self._last_jac_m = None
        self._last_jac_val = None

        
        super().__init__(len(v_mu), c)
        
    
    def record_qmc_stats(self):
        """Store selected damping parameters K in history trackers."""

        # ---- 1D (g) ----
        # keys are (i, side1d) with side1d ∈ {True, False}
        for key_1d, K in self._last_K1.items():
            if K is None:
                continue
            rec = self._history_1d[key_1d]  # defaultdict provides the dict
            rec["K"].append(-self._last_K1.get(key_1d))
        # ---- 2D (h) ----
        # keys are (i, j, sx, sy) with i<j and sx,sy ∈ {True, False}
        for key_2d, K_vec in self._last_K2.items():
            if K_vec is None:
                continue
            rec = self._history_2d[key_2d]
            rec["K"].append(-self._last_K2.get(key_2d))

    
    
    # Adaptive choice of Sobol sample size.
    def _divide_sobol(self,mult_factor):
        """Scale current Sobol sample size by `mult_factor` with minimum 1."""
       
        self._N_current= max(int(self._N_current * mult_factor),1)
    

    # QMC sequence by dimension (1D/2D) for component integrands.

    def _get_cached_sobol(self, N, m, multi_dim=False, seed_start=None):
        """Return digitally-shifted Sobol samples for 1D or 2D."""

        dim = 2 if multi_dim else 1
        cache_sobol = self._sobol_1d_cache if dim ==1 else self._sobol_2d_cache
        if seed_start is None:
            raise ValueError("seed_start must be provided so we can keep the sequences independent.")
        
        key = (int(N), int(m), dim) #int(seed_start)

        if key in cache_sobol:
            return cache_sobol[key][:, :N, :]
        # unshift Sobol sequence
        base = self._sobol_base.get(dim)
        if base is None or base.shape[0] < N:
            generator = qmcpy.DigitalNetB2(dim, graycode=True,randomize = None)  # no randomization
            base = np.asarray(generator.gen_samples(N))
            self._sobol_base[dim] = base
            base_bits = np.asarray(
                np.floor(base * (1 << self._DIGITAL_SHIFT_BITS)),
                dtype=np.uint64,
            )
            self._sobol_base_bits[dim] = base_bits
        base = self._sobol_base[dim][:N]
        base_bits = self._sobol_base_bits[dim][:N]

        arr = np.empty((m, N, dim), dtype=np.float64)
        # Digital shift randomization.
        for idx, seed in enumerate(range(seed_start, seed_start + m)):
            rng = np.random.default_rng(seed)
            shift_bits = rng.integers(
                0,
                1 << self._DIGITAL_SHIFT_BITS,
                size=dim,
                dtype=np.uint64,
            )
            shifted_bits = np.bitwise_xor(base_bits, shift_bits)
            arr[idx] = shifted_bits.astype(np.float64) * self._DIGITAL_SHIFT_SCALE
        cache_sobol[key] = arr.copy()
        return arr
    
    
    # ---------------------------------------------------------------------------
    # Domain-transformed covariance used for IS sampling.
    # ---------------------------------------------------------------------------
    def _rescale_cov(self, Sigma_block):
        Sigma_block = np.atleast_2d(Sigma_block)
        if Sigma_block.shape == (1, 1):
            sig_scale = self._epsilon * Sigma_block
            sig_tildle = np.linalg.inv(sig_scale)
            return float(sig_tildle[0, 0])
        else:
            sig_scale = self._epsilon * Sigma_block
            sig_tildle = np.linalg.inv(sig_scale)
        return sig_tildle
    
    # ---------------------------------------------------------------------------
    # Adaptive damping selection for component difference integrands.
    # ---------------------------------------------------------------------------
    def _select_K_1D(self, idx1, m_curr, side1d=True,m_prev = None):
        """
        Unified K-selection for:
        - 1D:  g_plus (K>beta), g_minus (K<beta)
        - 2D:  h_plus (K1>beta, K2>beta), h_minus (K1<beta, K2<beta)
        Returns:
        (K_scalar, elapsed) if idx1 set and idx2 is None
        (K_vec[2], elapsed) if idx1 and idx2 set (order matches inputs)
        """
        if m_prev is not None:
           m_curr = m_curr-m_prev
        # --- helpers -------------------------------------------------------------
        eps = 0.01
        lb = self._v_beta + eps if side1d else -np.inf
        ub = np.inf if side1d else self._v_beta - eps

        x0 = self._last_K1[(idx1, side1d)]

        if x0 is None:
            x0 = 1 if side1d else -1

        # --- 1D branch -----------------------------------------------------------

        sigma_i_sc = self._v_sigma[idx1, idx1]
        lam_da = 0

        def F1(K):
            # value
            val_loss, _, _ = self._loss_1D_grad_hess(K, side1d)
            val_cf, _, _ = self._log_cf_grad_hess_1d(K, idx1)
            return (
                val_loss
                + val_cf
                + 0.5 * lam_da * (K**2) * sigma_i_sc
                - K * m_curr
            )

        def G1(K):
            # gradient
            _, grad_loss, _ = self._loss_1D_grad_hess(K, side1d)
            _, grad_cf, _ = self._log_cf_grad_hess_1d(K, idx1)
            return grad_loss + grad_cf + lam_da * K * sigma_i_sc - m_curr


        bnds = Bounds(lb, ub, keep_feasible=True)

        res = minimize(fun=F1, x0=x0, method="SLSQP", jac=G1, bounds=bnds)

        K_sol = float(res.x[0])
        ok_1d = (K_sol > self._v_beta) if side1d else (K_sol < self._v_beta)
        # Feasibility guard (stick to side; if not, keep previous x0)
        if not ok_1d:
            K_sol = x0

        return K_sol


    def _select_K_2d(self, idx1, idx2, m_curr, side2d=(True, True), m_prev = None):

        if m_prev is not None:
            m_curr = m_curr-m_prev
        sx, sy = side2d
        eps = 0.01
        lb = np.array(
            [
                self._v_beta + eps if sx else -np.inf,
                self._v_beta + eps if sy else -np.inf,
            ]
        )
        ub = np.array(
            [np.inf if sx else self._v_beta - eps, np.inf if sy else self._v_beta - eps]
        )
        # canonical order in storage
        i, j = (int(idx1), int(idx2))
        if i > j:
            i, j = j, i

        key = (i, j, bool(sx), bool(sy))
        x0_vec = self._last_K2[key]

        if x0_vec is None:
            const = 1.0
            x0_vec = np.array([const if sx else -const, const if sy else -const])

        sigma_2d = self._v_sigma[np.ix_([i, j], [i, j])]
        lam_da = 0


        def F2(K_vec):
            # value
            val_loss, _, _ = self._loss_2D_grad_hess(K_vec, side2d)
            val_cf, _, _ = self._log_cf_grad_hess_2d(K_vec, idx1=i, idx2=j)
            quad = 0.5 * lam_da * float(K_vec @ (sigma_2d @ K_vec))
            return val_loss + val_cf + quad - float(K_vec @ m_curr)

        def G2(K_vec):
            # gradient
            _, g_loss, _ = self._loss_2D_grad_hess(K_vec, side2d)
            _, g_cf, _ = self._log_cf_grad_hess_2d(K_vec, idx1=i, idx2=j)
            return g_loss + g_cf + lam_da * (sigma_2d @ K_vec) - m_curr

            # (Optional) Hessian if you switch to trust-constr
            # def H2(K_vec):
            #     _, _, h_loss = self._loss_grad_hess_by_name(loss_name, K_vec, idx1=i, idx2=j)
            #     _, _, h_cf   = self._log_cf_grad_hess_2d(K_vec, idx1=i, idx2=j)
            #     return h_loss + h_cf + lam_da * sigma_2d

        bnds = Bounds(lb, ub, keep_feasible=True)

        res = minimize(fun=F2, x0=x0_vec, method="SLSQP", jac=G2, bounds=bnds)

        K_vec = np.array(res.x, dtype=float)

        # Feasibility guard: both components must lie on the requested side
        okx = (K_vec[0] > self._v_beta) if sx else (K_vec[0] < self._v_beta)
        oky = (K_vec[1] > self._v_beta) if sy else (K_vec[1] < self._v_beta)

        if not (okx and oky):
            K_vec = x0_vec
        # Put back in caller’s order
        '''
        if i > j:
            K_vec = K_vec[::-1]
        '''


        return K_vec

    
    # ---------------------------------------------------------------------------
    # Log-transformed loss and CF terms used in damping optimization.
    # ---------------------------------------------------------------------------
    def _log_cf_grad_hess_1d(self, K_scalar, idx):

        mu_i = self._v_mu[idx]
        sigma_ii = self._v_sigma[idx, idx]
        val = -K_scalar * mu_i + 0.5 * (K_scalar**2) * sigma_ii
        grad = -mu_i + sigma_ii * K_scalar
        hess = sigma_ii
        return val, grad, hess

    def _log_cf_grad_hess_2d(self, K_pair, idx1, idx2):
       
        mu_sub = self._v_mu[[idx1, idx2]]
        sigma_sub = self._v_sigma[np.ix_([idx1, idx2], [idx1, idx2])]
        val = -K_pair.T.dot(mu_sub) + 0.5 * K_pair.T @ sigma_sub @ K_pair
        grad = -mu_sub + sigma_sub.dot(K_pair)
        hess = sigma_sub
        return val, grad, hess

    def _loss_1D_grad_hess(self, K, side1d=True):

        if side1d:
            K_beta = K - self._v_beta
            sign_grad = -1.0
        else:
            K_beta = self._v_beta - K
            sign_grad = 1.0

        val = -np.log(K_beta)
        grad = sign_grad / K_beta
        hess = 1 / (K_beta * K_beta)

        return val, grad, hess

    def _loss_2D_grad_hess(self, K, side2d=(True, True)):
        sx, sy = bool(side2d[0]), bool(side2d[1])
        side = np.array([sx, sy])
        gap = np.where(side, K - self._v_beta, self._v_beta - K)
        sgn = np.where(side, -1.0, 1.0)

        inv_gap = 1 / gap
        val = (-np.log(gap)).sum()
        grad = sgn * inv_gap
        hess = np.diag((inv_gap * inv_gap))

        return val, grad, hess
    
    def _maybe_scalar(self,x):
        return x.item() if np.ndim(x) == 0 else x
    
    # ============================================================================
    # Numerical Fourier component terms
    # ============================================================================
    # Characteristic function for component integrands.
    def char_function(self, t, i=None, j=None):
        """Characteristic function for full, 1D marginal, or 2D pair inputs."""
        mu = self._v_mu           
        Sigma = self._v_sigma     
        t_arr = np.asarray(t)
        # Full CF
        if i is None and j is None:
         
            if t_arr.ndim == 1:
                t_arr = t_arr[None, ...]
            if t_arr.shape[-1] != mu.shape[0]:
                raise ValueError(f"t has last dim {t_arr.shape[-1]} but mu has dim {mu.shape[0]}")

        
            lin = 1j * np.einsum('...d,d->...', t_arr, mu, optimize=True)

       
            quad = np.einsum('...d,df,...f->...', t_arr, Sigma, t_arr, optimize=True)

            out = np.exp(lin - 0.5 * quad)
            return out[0] if out.shape[0] == 1 else out

        # 1D marginal CF.
        elif i is not None and j is None:
            ti = np.asarray(t) 
            mu_i = mu[i]
            sig_ii = Sigma[i, i]
            return self._maybe_scalar(np.exp(1j * mu_i * ti - 0.5 * sig_ii * ti**2))

        # 2D pairwise CF.
        elif i is not None and j is not None:
            tij = np.asarray(t)  
            if tij.shape[-1] != 2:
                raise ValueError("For pairwise CF, t must have last dimension 2 (i.e., shape (...,2)).")

          
            if i > j:
                i, j = j, i
                tij = tij[..., ::-1]  

            t1 = tij[..., 0]
            t2 = tij[..., 1]

            mu_i, mu_j = mu[i], mu[j]
            sig_ii = Sigma[i, i]
            sig_jj = Sigma[j, j]
            sig_ij = Sigma[i, j]

            lin = 1j * (mu_i * t1 + mu_j * t2)
            quad = sig_ii * t1**2 + 2.0 * sig_ij * t1 * t2 + sig_jj * t2**2
            return self._maybe_scalar(np.exp(lin - 0.5 * quad))

        else:
            raise ValueError("Invalid index combination for char_function.")

    
    # 1D component difference-integrand.
    def _phase_1D(self, u, m_val, K, m_prev=None):
        if m_prev is None:
            return np.exp(1j * u * m_val)

        delta = m_val - m_prev

        return (
            np.exp(1j * u * m_prev)
            * np.expm1((1j * u - K) * delta)
    )


    def g_fourier_integrand(
        self, u, idx, m_val, K, positive_sign=True, m_prev=None
    ):
        """Integrand for g-term, with outer damping factored out."""
        i = 1j
        ui_K = u + i * K

        phase = self._phase_1D(u, m_val, K, m_prev)
        cf = self.char_function(ui_K, idx)

        indic = (
            K - self._v_beta - i * u
            if positive_sign
            else self._v_beta - K + i * u
        )

        return phase * cf / indic


    def _compute_1d(self, idx, m_val, sign=True, m_prev = None,hess_comp = False):
        """Handles both g/f single‐term and CV‐difference if control_var=True."""
        # pick integrand & coefficient
        # key_1D = (loss_name,idx,float(m_val))
        if hess_comp:
            N_current = self._N_fixed
        else:
            N_current = self._N_current

        key_parts = ["g", idx, bool(sign), float(m_val)]

        if m_prev is not None:
            key_parts.append(float(m_prev))


        key_1D = tuple(key_parts)
        if key_1D in self._cache_1d_val:
            return self._cache_1d_val[key_1D], self._cache_1d_V_list[key_1D]

        K_1D = self._select_K_1D(idx, m_val,side1d=sign,m_prev = m_prev)
        m_val_scalar = float(np.asarray(m_val).item())
        if m_prev is None:
            seed_key = (idx, m_val_scalar, None)
        else:
            m_prev_scalar = float(np.asarray(m_prev).item())
            seed_key = (idx, m_val_scalar, m_prev_scalar)

        seed_start = self._sobol_seed_map_1d.get(seed_key) # At new keys the seed start always become None
        if seed_start is None:
            seed_start = self._sobol_seed_1d
            self._sobol_seed_map_1d[seed_key] = seed_start
            self._sobol_seed_1d += self._m_shift



        qmc_seq_1D = self._get_cached_sobol(
            N_current,
            self._m_shift,
            seed_start= seed_start,
        )

        coef = self._coeff_1D

        integrand_1D = partial(self.g_fourier_integrand, positive_sign=sign, m_prev = m_prev)


        # Estimate +side
        est, V_list = RQMC_Fourier_1D_MN_vec(
            K_1D,
            idx,
            m_val,
            m_prev,
            N_current,
            self._m_shift,
            sigma_IS=self._sigma_trans_1D[idx],
            integrand_func=integrand_1D,
            qmc_seq=qmc_seq_1D,
        )
        anchor = m_val if m_prev is None else m_prev
        scale = coef * np.exp(-K_1D * anchor)

        val_1D = scale * est
        V_list_1D = scale * V_list

        if hess_comp:
            return val_1D

        # record history & new K
        self._last_K1[(idx, bool(sign))] = K_1D

        self._cache_1d_val[key_1D] = val_1D
        self._cache_1d_V_list[key_1D] = V_list_1D

        return val_1D, V_list_1D


    def _compute_1d_2side(self, idx, m_val, m_prev=None, hess_comp=False):
        """Sum both 1D sides and aggregate per-shift samples."""
        if hess_comp:
            val_pos = self._compute_1d(idx, m_val, sign=True, m_prev=m_prev, hess_comp=True)
            val_neg = self._compute_1d(idx, m_val, sign=False, m_prev=m_prev, hess_comp=True)
            return val_pos + val_neg

        val_pos, V_pos = self._compute_1d(idx, m_val, sign=True, m_prev=m_prev, hess_comp=False)
        val_neg, V_neg = self._compute_1d(idx, m_val, sign=False, m_prev=m_prev, hess_comp=False)
        return val_pos + val_neg, V_pos + V_neg

    # 2D component difference integrands.
    def _phase_2D(self, u, v, x, y, K, xy_prev=None):
        # Accept either a scalar K or a pair.
        Kx, Ky = np.broadcast_to(
            np.asarray(K, dtype=float), (2,)
        )

        if xy_prev is None:
            return np.exp(1j * (u * x + v * y))

        x_prev, y_prev = xy_prev

        delta = (
            (1j * u - Kx) * (x - x_prev)
            + (1j * v - Ky) * (y - y_prev)
        )

        return (
            np.exp(1j * (u * x_prev + v * y_prev))
            * np.expm1(delta)
        )


    def h_fourier_integrand(
        self, u, v, idx1, idx2, x, y, K,
        positive_sign=(True, True), xy_prev=None
    ):
        """Integrand for h-term, with outer damping factored out."""
        i = 1j

        u, v = np.broadcast_arrays(
            np.asarray(u), np.asarray(v)
        )

        Kx, Ky = np.broadcast_to(
            np.asarray(K, dtype=float), (2,)
        )

        ui_K0 = u + i * Kx
        ui_K1 = v + i * Ky

        phase = self._phase_2D(
            u, v, x, y, (Kx, Ky), xy_prev
        )

        cf = self.char_function(
            np.stack([ui_K0, ui_K1], axis=-1),
            idx1,
            idx2,
        )

        sx, sy = bool(positive_sign[0]), bool(positive_sign[1])

        s_x = -i if sx else i
        s_y = -i if sy else i

        b_x = -self._v_beta if sx else self._v_beta
        b_y = -self._v_beta if sy else self._v_beta

        denom = (
            (s_x * ui_K0 + b_x)
            * (s_y * ui_K1 + b_y)
        )

        return phase * cf / denom



    def _compute_2d(self, i, j, m_val, sign2d=(True, True),m_prev = None,hess_comp = False):
        """Handles both h/l single‐term and CV‐difference if control_var=True."""

        m_val_2D = m_val[[i, j]]
        key_parts = [
            "h",
            int(i),
            int(j),
            bool(sign2d[0]),
            bool(sign2d[1]),
            tuple(m_val),
        ]

        if m_prev is not None:
            m_prev_2D = m_prev[[i,j]]
            key_parts.append(tuple(m_prev_2D))
        else:
            m_prev_2D = None

        key_2D = tuple(key_parts)

        if key_2D in self._cache_2d_val:
            return self._cache_2d_val[key_2D], self._cache_2d_V_list[key_2D]

        K_2D= self._select_K_2d(i, j,m_val_2D,side2d=sign2d,m_prev = m_prev_2D)
        key = (i,j)

        key_sort = (i,j) if i<j else (j,i)

        if hess_comp:
           N_current = self._N_fixed
        else:
           N_current = self._N_current


        idx_pair = key_sort
        order = list(idx_pair)
        m_pair_vals = tuple(np.asarray(m_val[order], dtype=float))
        if m_prev is None:
            m_prev_pair_vals = None
        else:
            m_prev_pair_vals = tuple(np.asarray(m_prev[order], dtype=float))

        seed_key = (idx_pair, m_pair_vals, m_prev_pair_vals)
        seed_start = self._sobol_seed_map_2d.get(seed_key)
        if seed_start is None:
            seed_start = self._sobol_seed_2d
            self._sobol_seed_map_2d[seed_key] = seed_start
            self._sobol_seed_2d += self._m_shift

        qmc_seq_2D = self._get_cached_sobol(
            N_current,
            self._m_shift,
            multi_dim=True,
            seed_start= seed_start,
        )
        


        coef = self._coeff_2D

        integrand_func_2D = partial(self.h_fourier_integrand, positive_sign=sign2d, xy_prev = m_prev_2D)


        est, V_list_2D = RQMC_Fourier_2D_MN_vec(
            K_2D,
            i,
            j,
            m_val_2D,
            m_prev_2D,
            N_current,
            self._m_shift,
            sigma_IS=self._sigma_trans_2D[key_sort],
            L_IS = self._L_2D[key_sort],
            integrand_func=integrand_func_2D,
            qmc_seq_2D=qmc_seq_2D,
        )
        

        anchor = (
            m_val_2D
            if m_prev_2D is None
            else m_prev_2D
        )

        Kx, Ky = np.broadcast_to(
            np.asarray(K_2D, dtype=float), (2,)
        )

        scale = coef * np.exp(
            -Kx * anchor[0]
            -Ky * anchor[1]
        )

        val_2D = scale * est

        if hess_comp:
            return val_2D

        V_list_2D = scale * V_list_2D


        key_hist = (min(i, j), max(i, j), bool(sign2d[0]), bool(sign2d[1]))
        self._last_K2[key_hist] = K_2D

        self._cache_2d_val[key_2D] = val_2D
        self._cache_2d_V_list[key_2D] = V_list_2D

        return val_2D, V_list_2D


    def _compute_2d_2side(self, i, j, m_val, m_prev=None, hess_comp=False):
        """Aggregate all four 2D sign combinations."""
        # List all four sign combinations.
        signs = [(True, True), (True, False), (False, True), (False, False)]

        if hess_comp:
            vals = [self._compute_2d(i, j, m_val, sign2d=s, m_prev=m_prev, hess_comp=True)
                    for s in signs]
            return sum(vals)

        vals, vars_ = zip(*[
            self._compute_2d(i, j, m_val, sign2d=s, m_prev=m_prev, hess_comp=False)
            for s in signs
        ])
        return sum(vals), sum(vars_)
    
    # ============================================================================
    # Aggregation all the Fourier components
    # ============================================================================
    # Loss aggregation: E[l(X - m)] from 1D and 2D components.
    def _loss_cv_samples(self, m, m_prev=None):
        """Loss baseline or CV increment: shape (m_shift,)."""
        d = len(m)
        samples = np.zeros(self._m_shift, dtype=float)

        for i in range(d):
            prev_i = None if m_prev is None else m_prev[i]

            _, vec = self._compute_1d_2side(
                i,
                m[i],
                m_prev=prev_i,
                hess_comp=False,
            )
            samples += vec

        if self._alpha:
            for i in range(d):
                for j in range(i + 1, d):
                    _, vec = self._compute_2d_2side(
                        i,
                        j,
                        m,
                        m_prev=m_prev,
                        hess_comp=False,
                    )
                    samples += self._alpha * vec

        return self._1_alpha * samples


    def _gradient_cv_samples(self, m, m_prev=None):
        """Gradient baseline or CV increment: shape (m_shift, d)."""
        d = len(m)
        samples = np.zeros((self._m_shift, d), dtype=float)

        for i in range(d):
            prev_i = None if m_prev is None else m_prev[i]

            _, vec = self._compute_1d_2side(
                i,
                m[i],
                m_prev=prev_i,
                hess_comp=False,
            )
            samples[:, i] += vec

            if self._alpha:
                for j in range(d):
                    if j == i:
                        continue

                    _, vec = self._compute_2d_2side(
                        j,
                        i,
                        m,
                        m_prev=m_prev,
                        hess_comp=False,
                    )
                    samples[:, i] += self._alpha * vec

        return self._1_alpha * self._v_beta * samples

    def _evaluate_cv_step(self, m, commit=False):
        """
        Evaluate the accumulated loss and positive loss gradient.

        Trial:
            Store one pending baseline/increment without changing history.

        Commit:
            Append the pending version and update the shared committed point.
        """
        m = self._check_argument(m)
        self._current_m = m.copy()

        m_prev = self._prev_m_cv

        if m_prev is not None and np.array_equal(m, m_prev):
            # Already committed: do not append the same version twice.
            self._pending_cv_step = None
            return float(self._accum_loss), self._accum_jac.copy()

        key = (
            tuple(m),
            None if m_prev is None else tuple(m_prev),
            int(self._N_current),
            int(self._m_shift),
        )

        step = self._pending_cv_step

        if step is None or step["key"] != key:
            step = {
                "key": key,
                "m": m.copy(),
                "m_prev": None if m_prev is None else m_prev.copy(),
                "N": int(self._N_current),
                "loss_samples": self._loss_cv_samples(m, m_prev),
                "gradient_samples": self._gradient_cv_samples(m, m_prev),
            }

        loss_value = float(
            self._accum_loss + step["loss_samples"].mean()
        )

        gradient_value = step["gradient_samples"].mean(axis=0)
        if self._accum_jac is not None:
            gradient_value = self._accum_jac + gradient_value

        if commit:
            self._cv_history.append(step)
            self._prev_m_cv = m.copy()

            self._accum_loss = loss_value
            self._accum_jac = gradient_value.copy()

            self._pending_cv_step = None
        else:
            # A different trial replaces the previous pending trial.
            self._pending_cv_step = step

        return loss_value, gradient_value.copy()

    def shortfall_risk(self, m, commit=False):
        loss_value, _ = self._evaluate_cv_step(m, commit=commit)
        return loss_value
    # Gradient-variance component of sandwich covariance V.
    def shortfall_risk_jac(self, m, commit=False):
        _, gradient_value = self._evaluate_cv_step(m, commit=commit)
        return gradient_value

    # ============================================================================
    # Statistical error components
    # ============================================================================

    def oracle_covariance(self, lam, loc_index=0):
        
        versions = list(self._cv_history)

        if self._pending_cv_step is not None:
            versions.append(self._pending_cv_step)

        if not versions:
            raise RuntimeError("No CV samples have been computed.")

        if loc_index is None:
            selected = versions[-1:]
        elif isinstance(loc_index, slice):
            selected = versions[loc_index]
        else:
            start = int(loc_index)

            if not -len(versions) <= start < len(versions):
                raise IndexError("Invalid covariance version index.")

            selected = versions[start:]

        if not selected:
            raise IndexError("The covariance selection is empty.")

        covariances = []

        for step in selected:
            oracle_samples = self.gradient_oracle(
                step["loss_samples"],
                step["gradient_samples"],
                lam,
                increment=step["m_prev"] is not None,
            )

            if oracle_samples.shape[0] < 2:
                raise ValueError("At least two shifts are required.")

            covariances.append(
                np.atleast_2d(
                    np.cov(oracle_samples, rowvar=False, ddof=1)
                )
            )

        return np.sum(covariances, axis=0)

    def gradient_oracle(
        self, loss_samples, gradient_samples, lam, increment=False
    ):
        """Baseline or increment oracle samples: shape (S, d+1)."""
        n_shifts, d = gradient_samples.shape

        oracle_samples = np.empty((n_shifts, d + 1), dtype=float)

        oracle_samples[:, :d] = -lam * gradient_samples
        oracle_samples[:, -1] = loss_samples

        # Constants occur only in the baseline.
        if not increment:
            oracle_samples[:, :d] += 1.0
            oracle_samples[:, -1] -= self.c

        return oracle_samples

    def mean_loss_hessian(self, m):
        """Direct mean Hessian at m; no CV accumulation or sample storage."""
        m = self._check_argument(m)
        self._current_m = m.copy()

        d = len(m)

        const_1 = self._1_alpha * self._v_beta**2
        const_2 = self._alpha * const_1

        hessian = np.zeros((d, d), dtype=float)

        for i in range(d):
            val = self._compute_1d_2side(
                i, m[i],
                m_prev=None,
                hess_comp=True,
            )
            hessian[i, i] = const_1 * val

        if self._alpha:
            for i in range(d):
                for j in range(i + 1, d):
                    val = self._compute_2d_2side(
                        i, j, m,
                        m_prev=None,
                        hess_comp=True,
                    )

                    contribution = const_2 * val

                    hessian[i, i] += contribution
                    hessian[j, j] += contribution
                    hessian[i, j] = contribution
                    hessian[j, i] = contribution

        return hessian

    def jacobian_oracle(self, m, lam, mean_gradient):
        """Mean oracle Jacobian with respect to (m, lambda)."""
        d = len(m)
        mean_hessian = self.mean_loss_hessian(m)

        jacobian = np.zeros((d + 1, d + 1), dtype=float)

        jacobian[:d, :d] = lam * mean_hessian
        jacobian[:d, -1] = -mean_gradient
        jacobian[-1, :d] = -mean_gradient

        return jacobian

    def stat_err_full(self, m, lam, alpha_conf=0.05, loc_index=0):
        """Marginal confidence half-widths for allocation components."""
        m = self._check_argument(m)
        d = len(m)

        # Full accumulated mean gradient at the requested point.
        _, mean_gradient = self._evaluate_cv_step(m, commit=False)

        # Covariance is computed per version, then summed.
        oracle_cov = self.oracle_covariance(
            lam, loc_index=loc_index
        )

        mean_jacobian = self.jacobian_oracle(
            m, lam, mean_gradient
        )

        left = np.linalg.solve(mean_jacobian, oracle_cov)
        sandwich = np.linalg.solve(mean_jacobian, left.T).T

        estimator_covariance = sandwich / self._m_shift

        critical_value = norm.ppf(1.0 - alpha_conf / 2.0)
        errors = critical_value * np.sqrt(
            np.diag(estimator_covariance)
        )

        return float(errors.max())

# ============================================================================
# End of module
# ============================================================================
