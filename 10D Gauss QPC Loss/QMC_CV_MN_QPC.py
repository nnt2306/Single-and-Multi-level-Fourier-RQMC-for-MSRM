import sys
import numpy as np
import time
import qmcpy
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from scipy.linalg import solve_triangular


import logging

logging.basicConfig(level=logging.INFO)


from msra_loss import CommonLossFunctionAbs

# ============================================================================
# RQMC estimators (Gaussian IS with domain transformation)
# Notes:
# - `m_prev` is included to evaluate difference-integrands across iterations.
# ============================================================================
def RQMC_Fourier_1D_MN_vec(K, idx, m_val,m_prev, N, m, sigma_IS, integrand_func, qmc_seq=None):
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

    # Map to proposal N(0, sigma_IS^2)
    U = norm.ppf(xi) * sigma_IS           


    F = integrand_func(U, idx, m_val, K,m_prev)
    

  
    quad = (U / sigma_IS)**2             
    log_IS = log_norm_const - 0.5*quad
    W = np.exp(-log_IS)                  

    V = np.real(F) * W                   
    V_shift_means = V.mean(axis=1)      
    return V_shift_means.mean(), V_shift_means



def RQMC_Fourier_2D_MN_vec(K, idx1, idx2, m_val, m_prev, N, m, sigma_IS,L_IS, integrand_func, qmc_seq_2D=None):
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
    
    F = integrand_func(U[...,0], U[...,1], idx1, idx2, m1, m2, K,m_prev)
    
    U2 = U.reshape(-1, 2)
    W = solve_triangular(L, U2.T, lower=True, overwrite_b=False).T 
    quad = (W**2).sum(axis=1).reshape(m, N)

    log_IS = log_norm_const - 0.5*quad
    V = np.real(F) * np.exp(-log_IS)
    V_shift_means = V.mean(axis=1)
    return V_shift_means.mean(), V_shift_means

# ============================================================================
# Multi-level / CV Fourier-RQMC class
# ============================================================================
class RQMC_CV_Fou_MN_qpc(CommonLossFunctionAbs):
    def __init__(
        self,
        v_mu,
        v_sigma,
        N_sobol = 64,
        m_shift=30,
        alpha=0,
        c=None,
        epsilon =1,
    ):
        """
        Initialize the multilevel RQMC Fourier solver for QPC loss.

        Component naming used throughout this class:
        - g: 1D loss component.
        - f: 1D gradient component (derivative of g contribution).
        - h: 2D loss interaction component.
        - l: 2D gradient interaction component (derivative of h contribution).
        - hess_1D / hess_2D: Hessian-related Fourier components.
        - hess_2D_boundary : Boundary correction term for 2D hessian


        Parameters:
        - v_mu: Mean vector.
        - v_sigma: Covariance matrix (or a vector reshaped into a matrix).
        - alpha: System-level coupling weight.
        - c: Optional inequality-constraint constant.
        - N_sobol: Number of Sobol points per shift (default: 64).
        - m_shift: Number of randomized digital shifts (default: 30).
        - epsilon: Scale used in domain transformation.
        """

        self._v_mu = np.array(v_mu)
        self._v_sigma = np.array(v_sigma).reshape((len(v_mu), len(v_mu)))
        self._alpha = alpha
        self._epsilon = epsilon
        
        
        self._N_fixed = int(N_sobol)
        self._N_current_loss= int(N_sobol)
        self._N_current_grad = int(N_sobol)
        self._m_shift = int(m_shift)

        d = len(self._v_mu)

        
        # Common constants for Fourier inversion
        self._coeff_f = -0.5 / np.pi
        self._coeff_g = 1.0 / np.pi
        self._coeff_hl = 1 / (2 * np.pi) ** 2

        self._current_m = None  

       # ---------------------------------------------------------------------------
        # Last damping parameters by component family.
        # For 1D terms (`g`, `f`, and Hessian helper component).
        # ---------------------------------------------------------------------------

        self._last_K1 = {
    loss: {i: None for i in range(d)}
    for loss in ("g","f","hess_1D")
}
        
    

        self._history_1d = {
            "g": {
                i: {"K": []}
                for i in range(d)
            },
            "f": {
                i: {"K": []}
                for i in range(d)
            },
        }
        self._cache_1d_val = {} 
        self._cache_1d_V_list = {}  
        # For 2D terms (`h`, `l`, and Hessian helper component).

        self._last_K2 = {
            loss: {(i,j): None for i in range(d) for j in range(i+1,d)}
            for loss in ("h","l","hess_2D")
        }


        self._history_2d = {
            "h": {
                (i, j): {"K": []}
                for i in range(d)
                for j in range(i + 1, d)
            },
            "l": {
                (i, j): {"K": []}
                for i in range(d)
                for j in range(i + 1, d)
            },
        }

        self._cache_2d_val = {}   # (loss_name, (i, j), tuple(m), [optional tuple(m_prev_ij)]) -> float
        self._cache_2d_V_list = {}

        self._cache_1d_K = {}
        self._cache_2d_K = {}

        self._sobol_1d_cache = {}
        self._sobol_2d_cache = {}

        # ---------------------------------------------------------------------------
        # Sobol backbone and shift bookkeeping.
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
                mat = self._rescale_cov(self._v_sigma[np.ix_([i, j], [i, j])])
                chol =  np.linalg.cholesky(mat)

          
                self._sigma_trans_2D[(i, j)] = mat
                self._L_2D[(i, j)] = chol

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
        
        self._accum_loss_samples = None
        self._accum_jac_samples = None

        self._cv_history = []
        self._pending_cv_step = None
        self._prev_m_cv = None

        self._accum_loss      = 0.0
        self._accum_jac       = None
        self._jac_components = []  
        self._pending_jac = None
        self._jac_cov_history = []
        self._pending_jac_cov = None


        # Memoized objective / gradient values.
        self._last_loss_m = None
        self._last_loss_val = None
        self._last_jac_m = None
        self._last_jac_val = None
        self._last_jac_cov = None

        
        super().__init__(len(v_mu), c)
           

    def record_qmc_stats(self):
        """Store selected damping parameters K in history trackers."""
        d = len(self._current_m)

        # 1-D terms
        for loss_name in ("g","f"):
            for i in range(d):
                rec = self._history_1d[loss_name][i]
                rec["K"].append(-self._last_K1[loss_name][i])

        if self._alpha:
            for loss_name in ("h","l"):
                for key in self._last_K2[loss_name]:
                    rec = self._history_2d[loss_name][key]
                    rec["K"].append(-self._last_K2[loss_name][key])

                   
    # Adaptive choice of Sobol sample size.
    def _divide_sobol(self,mult_factor ,grad = False):
        """Scale Sobol sample size for loss or gradient estimators with lower bound 1."""
        if grad:
            self._N_current_grad= max(int(self._N_current_grad * mult_factor),1)
        else:
            self._N_current_loss= max(int(self._N_current_loss * mult_factor),1
            )


        
    # QMC sequence by dimension (1D/2D) for component integrands.

    def _get_cached_sobol(self, N, m, multi_dim=False, seed_start=None):
        """Return cached digital-shift Sobol points with shape (m, N, dim)."""
        dim = 2 if multi_dim else 1
        cache_sobol = self._sobol_1d_cache if dim ==1 else self._sobol_2d_cache

        if seed_start is None:
            raise ValueError("seed_start must be provided so we can keep the sequences independent.")
        key = (int(N),int(m), dim, int(seed_start))

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
    # ---------------------------------------------------------------------------.

    def _rescale_cov(self,Sigma_block):
        """Return epsilon-scaled inverse covariance (scalar for 1D, matrix for 2D)."""
        Sigma_block = np.atleast_2d(Sigma_block)
        if Sigma_block.shape == (1, 1):
            sig_scale = self._epsilon * Sigma_block
            sig_tildle = np.linalg.inv(sig_scale)
            return float(sig_tildle[0,0])
        else:
            sig_scale = self._epsilon * Sigma_block
            sig_tildle = np.linalg.inv(sig_scale)
        return sig_tildle
    
    def _loss_grad_hess_by_name(self, name, K, idx1=None, idx2=None):
        """Dispatch to the loss/barrier value-gradient-Hessian by component name."""

        d = len(self._v_mu)

        # 1D component family.
        if name in ("g", "f","hess_1D"):
            if idx1 is None:
                raise ValueError(f"idx1 must be provided for loss '{name}'")
            val_1d,grad_1d, hess_1d = getattr(self, f"_loss_{name}_grad_hess")(K)
            return val_1d,grad_1d, hess_1d

        # 2D component family.
        if name in ("h", "l","hess_2D"):
            if idx1 is None or idx2 is None:
                raise ValueError(f"idx1 and idx2 must be provided for loss '{name}'")
            
            i,j = (idx1,idx2) if idx1<idx2 else (idx2,idx1)

            val2,grad2, hess2 = getattr(self, f"_loss_{name}_grad_hess")(K)
            

            return val2,grad2, hess2


        raise ValueError(f"Unknown loss name {name!r}")
    # ---------------------------------------------------------------------------
    # Adaptive optimal damping selection for component integrands
    # ---------------------------------------------------------------------------

    def _select_K(self, loss_name,m_curr, idx1=None, idx2=None,m_prev = None):


        """Select adaptive damping K for a 1D or 2D difference component family."""

        if m_prev is not None:
            m_curr = m_curr-m_prev
        
        # 1D branch (warm start + constrained optimization).
        if idx1 is not None and idx2 is None:
            i = idx1
            sigma_i_sc = self._v_sigma[i,i]
            lam_da = 0 ## Penalizing parameter for boundary-hugging control
            
            def F1_G1(K):
                val_loss,grad_loss,_= self._loss_grad_hess_by_name(loss_name, K, i)
                val_cf,grad_cf,_ = self._log_cf_grad_hess_1d(K,i)
                F1 = val_loss+val_cf + lam_da/2 *(K**2)*sigma_i_sc-K*(m_curr)
                G1 = (grad_loss+grad_cf)  + lam_da*K*sigma_i_sc-m_curr
                return  F1,G1

            
           
            if self._last_K1[loss_name][i] is None:
                K0 = 1
                self._last_K1[loss_name][i] = float(K0)
            
            eps = 0.01
            x0 = self._last_K1[loss_name][i]
            bnds = Bounds([eps],[np.inf], keep_feasible=True)

            res = minimize(
                fun= lambda K:F1_G1(K)[0],
                x0=x0,
                method="SLSQP",
                jac=lambda K:F1_G1(K)[1],
                bounds = bnds
            )
       
            K0 = float(res.x[0])

            # Feasibility guard.
            if K0 > 0:
                pass
            else:
                K0 = x0
             
            return K0

        # 2D branch (pairwise component).
        if idx1 is not None and idx2 is not None:
            i,j = (idx1,idx2) if idx1<idx2 else (idx2,idx1)
            sigma_2d = self._v_sigma[np.ix_([i,j],[i,j])]
            lam_da = 0
            
            def F2_G2(K_vec):
                val_loss,grad_loss,_ = self._loss_grad_hess_by_name(loss_name, K_vec, idx1=i,idx2=j)
                val_cf,grad_cf,_ = self._log_cf_grad_hess_2d(K_vec,idx1= i,  idx2 = j)
                F2 = val_loss+val_cf + lam_da/2 * float(np.dot(K_vec, sigma_2d @ K_vec))-K_vec.T.dot(m_curr)
                G2 = grad_loss+ grad_cf + lam_da*sigma_2d @ K_vec-m_curr
                return  F2,G2
            
           
            if self._last_K2[loss_name][(i, j)] is None:

                K0_vec = np.ones(2)
                
                self._last_K2[loss_name][(i, j)] = K0_vec.copy()

            eps = 0.01
            x0_vec = self._last_K2[loss_name][(i, j)]
            bnds = Bounds([eps,eps],[np.inf,np.inf], keep_feasible=True)

            res = minimize(
                fun=lambda K_vec:F2_G2(K_vec)[0],
                x0=x0_vec,
                method="SLSQP",
                jac=lambda K_vec:F2_G2(K_vec)[1],
                bounds = bnds
            )
            


            K0_vec = res.x.copy()
            # Feasibility guard.
            if np.all(K0_vec > 0):
                pass 
            else:
                K0_vec = x0_vec
            
            if idx1 > idx2:
                K0_vec = K0_vec[::-1]
            

            return K0_vec


    # ---------------------------------------------------------------------------
    # Log-transformed loss and CF terms used in damping optimization.
    # ---------------------------------------------------------------------------
    def _log_cf_grad_hess_1d(self, K_scalar, idx):
        mu_i = self._v_mu[idx]
        sigma_ii = self._v_sigma[idx, idx]
        val = -K_scalar*mu_i + 0.5 * (K_scalar**2)*sigma_ii
        grad = -mu_i + sigma_ii * K_scalar
        hess = sigma_ii
        return val,grad, hess


    def _log_cf_grad_hess_2d(self, K_pair, idx1, idx2):
        mu_sub = self._v_mu[[idx1, idx2]]
        sigma_sub = self._v_sigma[np.ix_([idx1, idx2], [idx1, idx2])]
        val = -K_pair.T.dot(mu_sub) + 0.5 * K_pair.T @ sigma_sub @ K_pair
        grad = -mu_sub + sigma_sub.dot(K_pair)
        hess = sigma_sub
        return val,grad, hess

    def _loss_g_grad_hess(self, K):
        val = -3 * np.log(K)
        grad = -3 / K
        hess = 3 / K**2
        return val,grad, hess

    def _loss_h_grad_hess(self, K):
        scar_vec = np.array([2.0, 2.0], dtype=float)
        val = -float(scar_vec.dot(np.log(K)))      # scalar

        grad = -scar_vec / K
        hess = np.diag(scar_vec / K**2)
        return val,grad, hess

    def _loss_f_grad_hess(self, K):
        val = -2* np.log(K)
        grad = -2 /K
        hess = 2/ K**2
        return val,grad, hess

    def _loss_l_grad_hess(self, K):
        scar_vec =  np.array([2.0, 1.0], dtype=float)

        val = -float(scar_vec.dot(np.log(K)))      # scalar

        grad = -scar_vec / K
        hess = np.diag(scar_vec / K**2)
        return val,grad, hess
    
    # Hessian helper barriers.
    def _loss_hess_1D_grad_hess(self, K):
        val = - np.log(K)
        grad = -1 /K
        hess = 1/ K**2
        return val,grad, hess
    
    def _loss_hess_2D_grad_hess(self, K):
        scar_vec =  np.array([1.0, 1.0], dtype=float)

        val = -float(scar_vec.dot(np.log(K)))      # scalar

        grad = -scar_vec / K
        hess = np.diag(scar_vec / K**2)
        return val,grad, hess


    def _maybe_scalar(self,x):
        return x.item() if np.ndim(x) == 0 else x
    
    # ============================================================================
    # Numerical Fourier integrands by component family
    # Mapping:
    # - e: linear component without Fourier transform
    # - g: 1D loss component
    # - f: 1D gradient component
    # - h: 2D loss component
    # - l: 2D gradient component
    # Difference-integrand mode is activated when `m_prev` / `xy_prev` is passed.
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
    

    def e(self,m):
        """Deterministic term: sum(mu - m)."""
        return np.sum(self._v_mu - m)


    # 1D components
    def _phase_1d(self, u, m_val, K, m_prev=None):
        """Phase after factoring out the outer damping exponential."""
        if m_prev is None:
            return np.exp(1j * u * m_val)

        delta = m_val - m_prev
        return (
            np.exp(1j * u * m_prev)
            * np.expm1((1j * u - K) * delta)
        )


    def g_fourier_integrand_vec(self, u, idx, m_val, K, m_prev=None):
        """Integrand for 1D loss component g."""
        ui_K = u + 1j * K
        phase = self._phase_1d(u, m_val, K, m_prev)
        cf = self.char_function(ui_K, idx)
        return phase * cf / (1j * ui_K**3)




    def f_fourier_integrand_vec(self, u, idx, m_val, K, m_prev=None):
        """Integrand for 1D gradient component f."""
        ui_K = u + 1j * K
        phase = self._phase_1d(u, m_val, K, m_prev)
        cf = self.char_function(ui_K, idx)
        return phase * cf / ui_K**2

    
    def hess_1D_fourier_integrand_vec(
        self, u, idx, m_val, K, m_prev=None):

        """Integrand for 1D Hessian helper component."""
        ui_K = u + 1j * K
        phase = self._phase_1d(u, m_val, K, m_prev)
        cf = self.char_function(ui_K, idx)
        return phase * cf / (1j * ui_K)


    def _compute_1d(self, loss_name, idx, m_val,m_prev = None):
        """
        Compute one 1D component family at index `idx`.

        loss_name in {"g", "f", "hess_1D"}:
        - g: 1D loss component
        - f: 1D gradient component
        - hess_1D: 1D Hessian helper
        If `m_prev` is provided, compute the difference-integrand increment.
        """
        integrand = getattr(self, f"{loss_name}_fourier_integrand_vec")

        key_damp_1d = (loss_name, idx)


        if loss_name == "g":
            coef   = self._coeff_g
            N_current = self._N_current_loss
     
        elif loss_name == 'f':
            coef  = self._coeff_f
            N_current = self._N_current_grad

        elif loss_name == "hess_1D":
            coef = self._coeff_f
            N_current = self._N_fixed
        else:
            raise ValueError(f"Unsupported loss name: {loss_name}")
        key_parts = [int(N_current),loss_name, idx, float(m_val)]

        if m_prev is not None:
            key_parts.append(float(m_prev))

        key_1D = tuple(key_parts)
        
        if key_1D in self._cache_1d_val:
            return self._cache_1d_val[key_1D],self._cache_1d_V_list[key_1D]
        
        # Select damping for current step.

        K_new = self._select_K(loss_name, m_val ,idx1=idx,m_prev = m_prev)

        m_val_scalar = float(np.asarray(m_val).item())
        if m_prev is None:
            seed_key = (idx, m_val_scalar, None)
        else:
            m_prev_scalar = float(np.asarray(m_prev).item())
            seed_key = (idx, m_val_scalar, m_prev_scalar)
        
        seed_start = self._sobol_seed_map_1d.get(seed_key)
        # Use independent seeds for 1D component integrands across levels.
        if seed_start is None:
            seed_start = self._sobol_seed_1d
            self._sobol_seed_map_1d[seed_key] = seed_start
            self._sobol_seed_1d += self._m_shift
        

        qmc_seq_1D = self._get_cached_sobol(
            N_current,
            self._m_shift,
            seed_start=seed_start,
        )
       
        
        est, V_list_1D = RQMC_Fourier_1D_MN_vec(
            K_new, idx,
            m_val,
            m_prev,
            N_current, self._m_shift,
            sigma_IS=self._sigma_trans_1D[idx],
            integrand_func=integrand,
            qmc_seq= qmc_seq_1D
        )


        anchor = m_val if m_prev is None else m_prev
        damping = np.exp(-K_new * float(np.asarray(anchor).item()))

        scale = coef * damping

        val_1D = scale * est
        V_list_1D = scale * V_list_1D

        if loss_name == "hess_1D":
            return val_1D

        # Record history and cache.
        self._last_K1[loss_name][idx]     = K_new

        self._cache_1d_val[key_1D] = val_1D
        self._cache_1d_V_list[key_1D] = V_list_1D

        return val_1D,V_list_1D

   

    # 2D components.

    def _phase_2d(self, u, v, x, y, K, xy_prev=None):
        """Phase with the outer damping exponential factored out."""
        if xy_prev is None:
            return np.exp(1j * (u * x + v * y))

        x_prev, y_prev = xy_prev

        delta = (
            (1j * u - K[0]) * (x - x_prev)
            + (1j * v - K[1]) * (y - y_prev)
        )

        return (
            np.exp(1j * (u * x_prev + v * y_prev))
            * np.expm1(delta)
        )


    def h_fourier_integrand_vec(
        self, u, v, idx1, idx2, x, y, K, xy_prev=None
    ):
        """Integrand for 2D loss component h."""
        ui_K0 = u + 1j * K[0]
        ui_K1 = v + 1j * K[1]

        phase = self._phase_2d(u, v, x, y, K, xy_prev)  # stable even when delta ≈ 0
        cf = self.char_function(
            np.stack([ui_K0, ui_K1], axis=-1), idx1, idx2
        )

        return phase * cf / (ui_K0**2 * ui_K1**2)


    def l_fourier_integrand_vec(
        self, u, v, idx1, idx2, x, y, K, xy_prev=None
    ):
        """Integrand for 2D gradient component l."""
        ui_K0 = u + 1j * K[0]
        ui_K1 = v + 1j * K[1]

        phase = self._phase_2d(u, v, x, y, K, xy_prev)  # stable even when delta ≈ 0
        cf = self.char_function(
            np.stack([ui_K0, ui_K1], axis=-1), idx1, idx2
        )

        return phase * cf / (1j * ui_K0**2 * ui_K1)

    
    def hess_2D_fourier_integrand_vec(
        self, u, v, idx1, idx2, x, y, K, xy_prev=None
    ):
        """Integrand for 2D Hessian helper component."""
        ui_K0 = u + 1j * K[0]
        ui_K1 = v + 1j * K[1]

        phase = self._phase_2d(u, v, x, y, K, xy_prev)
        cf = self.char_function(
            np.stack([ui_K0, ui_K1], axis=-1), idx1, idx2
        )

        return phase * cf / (ui_K0 * ui_K1)



    def hess_2D_boundary_fourier_integrand_vec(
        self, u, v, idx1, idx2, x, y, K, xy_prev=None
    ):
        '''Intergrand for boundary correction term'''

        boundary_K = (K[0], 0.0)
        ui_K0 = u + 1j * K[0]

        phase = self._phase_2d(
            u, v, x, y, boundary_K, xy_prev
        )
        cf = self.char_function(
            np.stack([ui_K0, v], axis=-1), idx1, idx2
        )

        return phase * cf / ui_K0**2


    def _compute_2d(self, loss_name, i, j, m_val,m_prev = None):
        """
        Compute one 2D component family for pair (i, j).

        loss_name in {"h", "l", "hess_2D","hess_2D_boundary"}:
        - h: 2D loss component
        - l: 2D gradient component
        - hess_2D: 2D Hessian helper
        - hess_2D_boundary: 2D Hessian for boundary correction term 
        If `m_prev` is provided, compute the difference-integrand increment.
        """
        
        if loss_name == 'h':
            N_current = self._N_current_loss
        elif loss_name == 'l':
            N_current = self._N_current_grad
        else:
            N_current = self._N_fixed
 
        m_val_2D = m_val[[i,j]]
        
        key = (i,j)
        key_sort = (i,j) if i<j else (j,i)
        key_parts = [int(N_current),loss_name,key,tuple(m_val)]

        key_damp_2d = (loss_name,key)

         
        key_qmc = key


        integrand = getattr(self, f"{loss_name}_fourier_integrand_vec")
        if m_prev is not None:
            m_prev_2D = m_prev[[i,j]]
            key_parts.append(tuple(m_prev_2D))
        else:
            m_prev_2D = None

        key_2D = tuple(key_parts)
      
        
        coef   = self._coeff_hl


        if key_2D in self._cache_2d_val:
            return self._cache_2d_val[key_2D],self._cache_2d_V_list[key_2D]
        
        
        if loss_name == 'hess_2D_boundary':
        # Select damping for current step.
            K_new = self._select_K(loss_name = 'hess_2D',m_curr = m_val_2D, idx1=i, idx2=j,m_prev = m_prev_2D)
        else:
            K_new = self._select_K(loss_name = loss_name,m_curr = m_val_2D, idx1=i, idx2=j,m_prev = m_prev_2D)

        idx_pair = key_sort
        order = list(idx_pair)
        m_pair_vals = tuple(np.asarray(m_val[order], dtype=float))
        if m_prev is None:
            m_prev_pair_vals = None
        else:
            m_prev_pair_vals = tuple(np.asarray(m_prev[order], dtype=float))
        
        seed_key = (idx_pair, m_pair_vals, m_prev_pair_vals)
        seed_start = self._sobol_seed_map_2d.get(seed_key)
        # Use independent seeds for 2D component integrands across levels.

        if seed_start is None:
            seed_start = self._sobol_seed_2d
            self._sobol_seed_map_2d[seed_key] = seed_start
            self._sobol_seed_2d += self._m_shift

        
        
        qmc_seq_2D = self._get_cached_sobol(
            N_current,
            self._m_shift,
            multi_dim=True,
            seed_start=seed_start,
        )
        
        

        est, V_list_2D = RQMC_Fourier_2D_MN_vec(
            K_new, i, j,
            m_val_2D,
            m_prev_2D,
            N_current, self._m_shift,
            sigma_IS= self._sigma_trans_2D[key_qmc],
            L_IS = self._L_2D[key_qmc],
            integrand_func=integrand,
            qmc_seq_2D= qmc_seq_2D
        )


        anchor = m_val_2D if m_prev_2D is None else m_prev_2D

        if loss_name == "hess_2D_boundary":
            damping = np.exp(-K_new[0] * anchor[0])
        else:
            damping = np.exp(
                -K_new[0] * anchor[0]
                -K_new[1] * anchor[1])

        scale = coef * damping
        val_2D = scale * est


        if loss_name in ("hess_2D", "hess_2D_boundary"):
            return val_2D

        V_list_2D = scale * V_list_2D

        self._last_K2[loss_name][key_sort]   = K_new if i<j else K_new[::-1]
        self._cache_2d_val[key_2D] = val_2D
        self._cache_2d_V_list[key_2D] = V_list_2D
        
        
        return val_2D, V_list_2D
    
    # ============================================================================
    # Aggregation all the Fourier components
    # ============================================================================
    # Loss aggregation: E[l(X - m)] from 1D and 2D components.

    def _loss_cv_samples(self, m, m_prev=None):
        """Loss baseline or increment: shape (m_shift,)."""
        d = len(m)

        # Linear term
        deterministic = self.e(m)
        if m_prev is not None:
            # e(m_k) - e(m_{k-1})
            deterministic -= self.e(m_prev)

        samples = np.full(
            self._m_shift, deterministic, dtype=float
        )

        #  1D terms
        for i in range(d):
            prev_i = None if m_prev is None else m_prev[i]

            _, vec_g = self._compute_1d(
                "g", i, m[i], m_prev=prev_i
            )
            samples += 0.5 * vec_g

        # 2D terms.
        if self._alpha:
            for i in range(d):
                for j in range(i + 1, d):
                    _, vec_h = self._compute_2d(
                        "h", i, j, m, m_prev=m_prev
                    )
                    samples += self._alpha * vec_h

        return samples


    def _gradient_cv_samples(self, m, m_prev=None):
        """Gradient baseline or increment: shape (m_shift, d)."""
        d = len(m)


        baseline_constant = 1.0 if m_prev is None else 0.0
        samples = np.full(
            (self._m_shift, d), baseline_constant, dtype=float
        )

        for i in range(d):
            prev_i = None if m_prev is None else m_prev[i]

            _, vec_f = self._compute_1d(
                "f", i, m[i], m_prev=prev_i
            )
            samples[:, i] += vec_f

            if self._alpha:
                for j in range(d):
                    if j == i:
                        continue

                    _, vec_l = self._compute_2d(
                        "l", j, i, m, m_prev=m_prev
                    )
                    samples[:, i] += self._alpha * vec_l

        return samples


    def _evaluate_cv_step(self,m,commit = False):
        m = self._check_argument(m)
        self._current_m = m.copy()

        m_prev = self._prev_m_cv

        if m_prev is not None and np.array_equal(m,m_prev):
            self._pending_cv_step = None
            return float(self._accum_loss), self._accum_jac.copy()


        key = (
            tuple(m),
            None if m_prev is None else tuple(m_prev),
            self._N_current_loss,
            self._N_current_grad,
        )

        step = self._pending_cv_step

        if step is None or step["key"] != key:
            step = {
                "key": key,
                "m": m.copy(),
                "m_prev": None if m_prev is None else m_prev.copy(),
                "loss_samples": self._loss_cv_samples(m, m_prev),
                "gradient_samples": self._gradient_cv_samples(m, m_prev)}


        loss_value = self._accum_loss + step["loss_samples"].mean()

        gradient_value = step["gradient_samples"].mean(axis=0)

        if self._accum_jac is not None:
            gradient_value = self._accum_jac + gradient_value

        if commit:
            self._cv_history.append(step)
            self._prev_m_cv = m.copy()

            self._accum_loss = float(loss_value)
            self._accum_jac = gradient_value.copy()

            self._pending_cv_step = None
        else:

            self._pending_cv_step = step

        return float(loss_value), gradient_value.copy()


    def shortfall_risk(self, m, commit=False):
        """Aggregate loss using baseline or CV-difference updates."""
        loss_value, _ = self._evaluate_cv_step(m, commit=commit)
        return loss_value



    # Loss gradient aggregation: E[∇l(X - m)] from 1D and 2D components

    def shortfall_risk_jac(self, m, commit=False):
        """Aggregate gradient using baseline or CV-difference updates."""
        _, gradient_value = self._evaluate_cv_step(m, commit=commit)
        return gradient_value

    # ============================================================================
    # Statistical error components
    # ============================================================================
    # Gradient-variance component of sandwich covariance V.
    def gradient_oracle(
        self, loss_samples, gradient_samples, lam, increment=False
    ):
        """One baseline or increment oracle vector per shift."""
        n_shifts, d= gradient_samples.shape
        oracle_samples = np.empty((n_shifts, d+1),dtype = float)

        oracle_samples[:,:d] = -lam * gradient_samples
        oracle_samples[:,-1] = loss_samples

        if not increment:
            oracle_samples[:, :d] += 1.0
            oracle_samples[:, -1] = self.c

        return oracle_samples
 

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


    def mean_loss_hessian(self, m):
        """Estimate E[Hessian l(X-m)], shape (d, d)."""
        m = self._check_argument(m)
        self._current_m = m.copy()
        d = len(m)

        hessian = np.zeros((d, d), dtype=float)

        for i in range(d):
            val = self._compute_1d('hess_1D', i, m[i])
            hessian[i, i] = val

            if self._alpha:
                for j in range(d):
                    if j == i:
                        continue

                    val_cross = self._compute_2d(
                        'hess_2D', j, i, m
                    )
                    hessian[i, j] -= self._alpha * val_cross


                    val_boundary = self._compute_2d(
                        'hess_2D_boundary', j, i, m)

                    hessian[i, i] -= self._alpha * val_boundary

        return hessian


    def stat_err_full(self, m, lam, alpha_conf=0.05, loc_index=0):
        m = self._check_argument(m)
        d = len(m)

        _, mean_gradient = self._evaluate_cv_step(m, commit=False)


        oracle_cov = self.oracle_covariance(
            lam, loc_index=loc_index
        )

        mean_hessian = self.mean_loss_hessian(m)

        mean_jacobian = np.zeros((d + 1, d + 1), dtype=float)
        mean_jacobian[:d, :d] = lam * mean_hessian
        mean_jacobian[:d, -1] = -mean_gradient
        mean_jacobian[-1, :d] = -mean_gradient

        left = np.linalg.solve(mean_jacobian, oracle_cov)
        sandwich = np.linalg.solve(mean_jacobian, left.T).T

        estimator_covariance = sandwich / self._m_shift

        critical_value = norm.ppf(1.0 - alpha_conf / 2.0)
        stat_errs = critical_value * np.sqrt(
            np.diag(estimator_covariance)
        )

        return float(stat_errs.max())

# ============================================================================
# End of module
# ============================================================================