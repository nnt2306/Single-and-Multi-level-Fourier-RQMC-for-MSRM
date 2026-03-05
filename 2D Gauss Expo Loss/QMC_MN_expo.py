import sys
import numpy as np
import qmcpy
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from scipy.linalg import solve_triangular

from functools import partial
from collections import defaultdict

path_pjt = "/Users/Nguye071/Documents/GitHub/Single-and-Multi-level-Fourier-RQMC-for-MSRM"
sys.path.insert(0, path_pjt)
from msra_loss import CommonLossFunctionAbs


# ============================================================================
# RQMC estimators (Gaussian IS with domain transformation)
# ============================================================================

# 1D estimator
def RQMC_Fourier_1D_MN_vec(K, idx, m_val, N, m, sigma_IS, integrand_func, qmc_seq=None):
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

    F = integrand_func(U, idx, m_val, K)

    quad = (U / sigma_IS)**2              
    log_IS = log_norm_const - 0.5*quad
    W = np.exp(-log_IS)                 

    V = np.real(F) * W                   
    V_shift_means = V.mean(axis=1)   
    return V_shift_means.mean(), V_shift_means

# 2D estimator
def RQMC_Fourier_2D_MN_vec(K, idx1, idx2, m_val, N, m, sigma_IS,L_IS, integrand_func, qmc_seq_2D=None):
    """Compute the 2D RQMC Fourier estimator and per-shift means."""
    m1,m2 = m_val
    
    L = np.asarray(L_IS)
    log_norm_const = -np.log(2*np.pi) - 0.5*np.log(np.linalg.det(sigma_IS))

    # (m, N, 2) Sobol uniforms
    if qmc_seq_2D is None:
        xi = np.empty((m, N, 2), dtype=float)
        for s in range(m):
            xi[s] = qmcpy.DigitalNetB2(2, graycode=True, randomize="DS", seed=s).gen_samples(N)
    else:
        xi = np.asarray(qmc_seq_2D, dtype=float).reshape(m, N, 2)

    # Map to N(0, Sigma)
    Z = norm.ppf(xi)             
    U = Z @ L.T              

    
    F = integrand_func(U[...,0], U[...,1], idx1, idx2, m1, m2, K)   

    
    U2 = U.reshape(-1, 2)
    W = solve_triangular(L, U2.T, lower=True, overwrite_b=False).T  
    quad = (W**2).sum(axis=1).reshape(m, N)

    log_IS = log_norm_const - 0.5*quad
    V = np.real(F) * np.exp(-log_IS)      
    V_shift_means = V.mean(axis=1)
    return V_shift_means.mean(), V_shift_means

# ============================================================================
# Single-level Fourier-RQMC class
# ============================================================================
class RQMC_Fou_MN_expo(CommonLossFunctionAbs):
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
        Initialize the single-level RQMC Fourier solver for exponential loss.

        Parameters:
        - v_mu: Mean vector.
        - v_sigma: Covariance matrix (or a vector reshaped into a matrix).
        - v_beta: Risk-aversion coefficient.
        - N_sobol: Number of Sobol points per shift (default: 64).
        - m_shift: Number of randomized digital shifts (default: 32).
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

        self._last_K2 = defaultdict(lambda: None)  # key: (i, j, sx, sy) with i < j


        self._history_2d = defaultdict(
            lambda: {"K": []}
        )

        self._cache_2d_val = {}
        self._cache_2d_V_list = {}
        # ---------------------------------------------------------------------------
        # Sobol backbone and shift bookkeeping for digital-shift randomization.
        # ---------------------------------------------------------------------------
        self._sobol_1d_cache = {}
        self._sobol_2d_cache = {}
        self._DIGITAL_SHIFT_BITS = 52
        self._DIGITAL_SHIFT_SCALE = 1.0 / (1 << self._DIGITAL_SHIFT_BITS)
        self._sobol_base = {}
        self._sobol_base_bits = {}

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
                chol = np.linalg.cholesky(mat)

                # store canonical key
                self._sigma_trans_2D[(i, j)] = mat
                self._L_2D[(i, j)] = chol

                # store reversed view
                self._sigma_trans_2D[(j, i)] = mat[np.ix_(swap, swap)]
                self._L_2D[(j, i)] = chol[np.ix_(swap, swap)]
        
        self._jac_components = []  # Holds per-shift samples for variance estimation.


        super().__init__(len(v_mu), c)
        

    def record_qmc_stats(self):
        """Store selected damping parameters K in history trackers."""

        # ---- 1D (g) ----
        # keys are (i, side1d) with side1d ∈ {True, False}
        for key_1d, K in self._last_K1.items():
            if K is None:
                continue
            rec = self._history_1d[key_1d]  # defaultdict provides the dict
            rec["K"].append(self._last_K1.get(key_1d))
        # ---- 2D (h) ----
        # keys are (i, j, sx, sy) with i<j and sx,sy ∈ {True, False}
        for key_2d, K_vec in self._last_K2.items():
            if K_vec is None:
                continue
            rec = self._history_2d[key_2d]
            rec["K"].append(self._last_K2.get(key_2d))
            
    # QMC sequence cache by dimension (1D/2D) for component integrands.
    def _get_cached_sobol(self, N, m, multi_dim=False, seed_start=None):
        """Return cached digitally-shifted Sobol samples for 1D or 2D.
        Generated once for each dimension
        """
        dim = 2 if multi_dim else 1
        cache_sobol = self._sobol_1d_cache if dim ==1 else self._sobol_2d_cache

        if seed_start is None:
            raise ValueError("seed_start must be provided so we can keep the sequences independent.")
        key = (int(N), int(m), dim, int(seed_start))

        if key in cache_sobol:
            return cache_sobol[key][:, :N, :]

        # Ensure deterministic backbone is long enough.
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
        """Return epsilon-scaled inverse covariance (scalar for 1D, matrix for 2D)."""
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
    # Adaptive damping selection for component integrands.
    # ---------------------------------------------------------------------------
    def _select_K_1D(self, idx1,m_curr,side1d=True):
        """
        Select 1D damping parameter K under the side constraint.

        side1d=True  -> K > beta
        side1d=False -> K < beta
        """

        eps = 0.01
        lb = self._v_beta + eps if side1d else -np.inf
        ub = np.inf if side1d else self._v_beta - eps

        x0 = self._last_K1[(idx1, side1d)]

        if x0 is None:
            x0 = 1 if side1d else -1

        # --- 1D branch -----------------------------------------------------------

        sigma_i_sc = self._v_sigma[idx1, idx1]
        lam_da = 0.0 ## Penalizing parameter for boundary-hugging control


        def F1(K):
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
            return grad_loss + grad_cf + lam_da * K * sigma_i_sc - m_curr  # Penalizing term.

        bnds = Bounds(lb, ub, keep_feasible=True)

        res = minimize(fun=F1, x0=x0, method="SLSQP", jac=G1, bounds=bnds)
       

        K_sol = float(res.x)
        ok_1d = (K_sol > self._v_beta) if side1d else (K_sol < self._v_beta)
        # Feasibility guard (stick to side; if not, keep previous x0)
        if not ok_1d:
            K_sol = x0

        return K_sol

    def _select_K_2d(self, idx1, idx2, m_curr, side2d=(True, True)):
        """Select 2D damping vector K under per-component side constraints."""
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
        lam_da = 0.0 ## Penalizing parameter for boundary-hugging control

        def F2(K_vec):
            # value
            val_loss, _, _ = self._loss_2D_grad_hess(K_vec, side2d)
            val_cf, _, _ = self._log_cf_grad_hess_2d(K_vec, idx1=i, idx2=j)
            quad = 0.5 * lam_da * float(K_vec @ (sigma_2d @ K_vec))  # Penalizing term.
            return val_loss + val_cf + quad - float(K_vec @ m_curr)

        def G2(K_vec):
            # gradient
            _, g_loss, _ = self._loss_2D_grad_hess(K_vec, side2d)
            _, g_cf, _ = self._log_cf_grad_hess_2d(K_vec, idx1=i, idx2=j)
            return g_loss + g_cf + lam_da * (sigma_2d @ K_vec) - m_curr


        bnds = Bounds(lb, ub, keep_feasible=True)

        res = minimize(fun=F2, x0=x0_vec, method="SLSQP", jac=G2, bounds=bnds)

        K_vec = np.array(res.x, dtype=float)

        # Feasibility guard: both components must lie on the requested side
        okx = (K_vec[0] > self._v_beta) if sx else (K_vec[0] < self._v_beta)
        oky = (K_vec[1] > self._v_beta) if sy else (K_vec[1] < self._v_beta)

        if not (okx and oky):
            K_vec = x0_vec

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

   

    # 1D component integrands.
    def g_fourier_integrand(self, u, idx, m_val, K, positive_sign=True):
        i = 1j
        ui_K = u + i * K
        phase = np.exp(-(K - i*u) * m_val)    # exp(-K m) * exp(-i u m)
        cf = self.char_function(ui_K, idx)


        indic = (
            (K - self._v_beta - u * i) if positive_sign else (self._v_beta - K + u * i)
        )


        return phase * cf / indic

    def _compute_1d(self, idx, m_val, sign=True):
        """Compute one-sided 1D component value and RQMC shift samples."""

        key = ("g", bool(sign), float(m_val))
        if key in self._cache_1d_val:
            return self._cache_1d_val[key], self._cache_1d_V_list[key]
        
        # Select damping for current step.

        K_1D = self._select_K_1D(idx, m_val, side1d=sign)
        
        qmc_seq_1D = self._get_cached_sobol(
            self._N_current,
            self._m_shift,
            seed_start= 0
        )

        coef = self._coeff_1D
        integrand_1D = partial(self.g_fourier_integrand, positive_sign=sign)
        

        # Estimate one side.
        est, V_list = RQMC_Fourier_1D_MN_vec(
            K_1D,
            idx,
            m_val,
            self._N_current,
            self._m_shift,
            sigma_IS=self._sigma_trans_1D[idx],
            integrand_func=integrand_1D,
            qmc_seq=qmc_seq_1D,
        )
        self._last_K1[(idx, bool(sign))] = K_1D
        val = coef * est

        self._cache_1d_val[key] = val
        self._cache_1d_V_list[key] = V_list

        return val, V_list

    def _compute_1d_2side(self, idx, m_val):
        """Sum both 1D sides and aggregate per-shift samples."""
        val_pos, V0 = self._compute_1d(idx, m_val, sign=True)
        val_neg, V1 = self._compute_1d(idx, m_val, sign=False)
        return val_pos + val_neg, V0 + V1

    # 2D component integrands.

    def h_fourier_integrand(self, u, v, idx1, idx2, x, y, K, positive_sign=(True, True)
):
        i = 1j

        # Broadcast u and v
        u = np.asarray(u)
        v = np.asarray(v)
        u, v = np.broadcast_arrays(u, v)

        # Handle K as scalar or pair
        K = np.asarray(K)
        if K.ndim == 0:
            Kx = Ky = K
        else:
            if K.shape[0] != 2:
                raise ValueError("K must be scalar or length-2 array-like.")
            Kx, Ky = K[0], K[1]

        # Complex shifts
        ui_K0 = u + i * Kx
        ui_K1 = v + i * Ky

        # Phase term: exp(-(Kx - i u) x - (Ky - i v) y)
        phase = np.exp(-(Kx - i*u) * x - (Ky - i*v) * y)

        # Characteristic function
        tij = np.stack([ui_K0, ui_K1], axis=-1)        
        cf  = self.char_function(tij, idx1, idx2)      

        # Side selection and component-wise betas
        sx, sy = bool(positive_sign[0]), bool(positive_sign[1])
        

        s_x = (-i) if sx else (i)          # ±i multiplier
        s_y = (-i) if sy else (i)
        b_x = -self._v_beta if sx else self._v_beta   # ±β_i
        b_y = -self._v_beta if sy else self._v_beta

   
        denom = (s_x * ui_K0 + b_x) * (s_y * ui_K1 + b_y)

        return (phase * cf) / denom


    def _compute_2d(self, i, j, m_val, sign2d=(True, True)):
        """Compute one 2D sign-combination value and RQMC shift samples."""

        m_val_2D = m_val[[i, j]]
        key = (
            "h",
            int(i),
            int(j),
            bool(sign2d[0]),
            bool(sign2d[1]),
            tuple(m_val_2D.tolist()),
        )

        if key in self._cache_2d_val:
            return self._cache_2d_val[key], self._cache_2d_V_list[key]
        
        # Select damping for current step.

        K_2D = self._select_K_2d(idx1=i, idx2=j, m_curr = m_val_2D,side2d=sign2d)
        key_sort = (i,j) if i<j else (j,i)
        
        
        qmc_seq_2D = self._get_cached_sobol(
            self._N_current,
            self._m_shift,
            multi_dim=True,
            seed_start=0
        )
        
        
        coef = self._coeff_2D

        integrand_func_2D = partial(self.h_fourier_integrand, positive_sign=sign2d)


        est, V_list = RQMC_Fourier_2D_MN_vec(
            K_2D,
            i,
            j,
            m_val_2D,
            self._N_current,
            self._m_shift,
            sigma_IS=self._sigma_trans_2D[key_sort],
            L_IS = self._L_2D[key_sort],
            integrand_func=integrand_func_2D,
            qmc_seq_2D=qmc_seq_2D,
        )

        key_hist = (min(i, j), max(i, j), bool(sign2d[0]), bool(sign2d[1]))
        self._last_K2[key_hist] = K_2D
    
        val_2D = coef * est
        self._cache_2d_val[key] = val_2D
        self._cache_2d_V_list[key] = V_list

        return val_2D, V_list

    def _compute_2d_2side(self, i, j, m_val):
        """Aggregate all four 2D sign combinations."""
        val_pos_11, V11 = self._compute_2d(i, j, m_val, sign2d=(True, True))
        val_pos_12, V12 = self._compute_2d(i, j, m_val, sign2d=(True, False))
        val_neg_11, V22 = self._compute_2d(i, j, m_val, sign2d=(False, False))
        val_neg_12, V21 = self._compute_2d(i, j, m_val, sign2d=(True, False))
        val_tol = val_pos_11 + val_pos_12 + val_neg_11 + val_neg_12
        V_tol = V11 + V12 + V22 + V21
        return val_tol, V_tol
    
    # ============================================================================
    # Aggregation all the Fourier components
    # ============================================================================
    # Loss aggregation: E[l(X - m)] from 1D and 2D components.
    def shortfall_risk(self, m):
        m = self._check_argument(m)

        d = len(m)

       
        g = sum(self._compute_1d_2side(i, m[i])[0] for i in range(d))
        h = 0.0
        if self._alpha:
            h = sum(
                self._compute_2d_2side(i, j, m)[0]
                for i in range(d)
                for j in range(i + 1, d)
            )
        return self._1_alpha * (g + (self._alpha * h))        
    
    # Loss gradient aggregation: E[∇l(X - m)] from 1D and 2D components.
    def shortfall_risk_jac(self, m):
        m = self._check_argument(m)

        d = len(m)

        jac = np.zeros_like(m, dtype=float)
        samples_jac = np.zeros((self._m_shift, d), dtype=float)

        for i in range(d):
                val,vec = self._compute_1d_2side(i, m[i])
                jac[i] = val
                samples_jac[:,i] = vec
                if self._alpha:
                    for j in range(d):
                        if j != i:
                            val_l,vec_l = self._compute_2d_2side(j, i, m)
                            jac[i] += self._alpha * val_l
                            samples_jac[:,i] += self._alpha * vec_l
                        
        samples_jac *= self._1_alpha * self._v_beta
        
        self._pending_jac = samples_jac              

    
        return self._1_alpha * self._v_beta * jac
    
    # ============================================================================
    # Statistical error components
    # ============================================================================

    # Gradient-variance component of sandwich covariance V.
    def shortfall_risk_jac_var(self):
        comps = self._jac_components
        if not comps:
            raise RuntimeError("shortfall_risk_jac_var called before any gradient components were recorded")

                
        self._var_per_comp = [np.cov(comp, rowvar=False, ddof=1) for comp in comps]
        return self._var_per_comp[-1]
    
    # Inverse Hessian component of sandwich covariance V.
    def shortfall_risk_hess_inv(self, m):
        d = len(m)
        const_1 = self._1_alpha * self._v_beta**2
        const_2 = self._alpha * self._v_beta**2 * self._1_alpha

        hess = np.zeros((d, d), dtype=float)

        global_comp = 0.0
        for i in range(d):
            for j in range(d):
                if j == i:
                    continue
                global_comp += const_2 * self._compute_2d_2side(j, i, m)[0]

        for i in range(d):
            hess[i, i] = const_1 * self._compute_1d_2side(i, m[i])[0] + global_comp
            for j in range(d):
                if j != i:
                    hess[i, j] = global_comp
        hess_inv = np.linalg.inv(hess)
        
        return hess_inv
    
    # Maximum diagonal-based CI radius for the solution estimate.
    def statistical_error_sol_RQMC(self, m, alpha_conf=0.05):
        d = len(m)
        var_jac = self.shortfall_risk_jac_var()
        inv_H = self.shortfall_risk_hess_inv(m)
        var_sol = inv_H @ var_jac @ inv_H 

        C_alpha = norm.ppf(1 - alpha_conf / 2)

        RQMC_stat_error = []

        for i in range(d):
            std_i = np.sqrt(var_sol[i, i])
            rqmc_err = C_alpha * std_i / np.sqrt(self._m_shift)
            RQMC_stat_error.append(rqmc_err)

        return np.max(RQMC_stat_error)

        
# ============================================================================
# End of module
# ============================================================================
