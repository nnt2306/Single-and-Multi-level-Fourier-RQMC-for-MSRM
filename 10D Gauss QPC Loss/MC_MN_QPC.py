import sys
import numpy as np
import time
import qmcpy
from scipy.stats import norm, multivariate_normal
from joblib import Parallel, delayed

from scipy.optimize import minimize, Bounds
from functools import partial

import logging
logging.basicConfig(level=logging.INFO)

from msra_loss import CommonLossFunctionAbs


# ============================================================================
# MC point generation and Fourier estimators
# ============================================================================

def generate_mc_points(
    N: int,
    seed=None,
):
    """Generate reproducible, independent 1D and 2D MC uniform samples."""
    if N <= 0:
        raise ValueError("N must be strictly positive.")

    # Independent child streams preserve reproducibility across dimensions.
    root_seed = np.random.SeedSequence(seed)
    seed_1d, seed_2d = root_seed.spawn(2)

    rng_1d = np.random.default_rng(seed_1d)
    rng_2d = np.random.default_rng(seed_2d)

    points_1d = rng_1d.random(N)
    points_2d = rng_2d.random((N, 2))

    return {
        1: points_1d,
        2: points_2d,
    }

def MC_Fourier_1D_N_vec(
    K,
    idx,
    m_val,
    integrand_func,
    U_1D,
    importance_weights,
):
    """Compute the 1D MC Fourier estimator and per-sample values."""
    U_1D = np.asarray(
        U_1D,
        dtype=float,
    )

    importance_weights = np.asarray(
        importance_weights,
        dtype=float,
    )

    if U_1D.shape != importance_weights.shape:
        raise ValueError(
            "U_1D and importance_weights must have "
            f"the same shape; got {U_1D.shape} and "
            f"{importance_weights.shape}."
        )

    
    integrand_values = integrand_func(
        U_1D,
        idx,
        m_val,
        K,
    )

    if integrand_values.shape != U_1D.shape:
        raise ValueError(
            "The integrand output must have shape "
            f"{U_1D.shape}, but got {integrand_values.shape}."
        )

    V = np.real(integrand_values) * importance_weights



    return np.mean(V),V


def MC_Fourier_2D_N_vec(
    K,
    idx1,
    idx2,
    m_val,
    integrand_func,
    U_2D,
    importance_weights
):
    """Compute the 2D MC Fourier estimator and per-sample values."""

    m1, m2 = m_val
    U_2D = np.asarray(U_2D,dtype = float)

    importance_weights = np.asarray(
        importance_weights,
        dtype=float,
    )

    if U_2D.ndim not in (2, 3) or U_2D.shape[-1] != 2:
        raise ValueError(
            "U_2D must have shape (N, 2) or "
            f"(N, S_shift, 2); got {U_2D.shape}."
        )

    expected_weight_shape = U_2D.shape[:-1]

    if importance_weights.shape != expected_weight_shape:
        raise ValueError(
            "importance_weights must have shape "
            f"{expected_weight_shape}; "
            f"got {importance_weights.shape}."
        )

    F = integrand_func(
        U_2D[..., 0],
        U_2D[..., 1],
        idx1,
        idx2,
        m1,
        m2,
        K,
    )

    V = np.real(F) * importance_weights
    
    return np.mean(V),V


# ============================================================================
# Single-level Fourier-MC class
# ============================================================================

class MC_Fou_MN_qpc(CommonLossFunctionAbs):
    def __init__(
        self,
        v_mu,
        v_sigma,
        N_mc = 64,
        alpha=0,
        c=None,
        epsilon =1,
        fix_damping = False,
        seed = 0,
    ):
        """
        Initialize the single-level MC Fourier solver for QPC.

        Component naming used throughout this class:
        - g: 1D loss component.
        - f: 1D gradient component (derivative of g contribution).
        - h: 2D loss interaction component.
        - l: 2D gradient interaction component (derivative of h contribution).
        - hess_1D / hess_2D: Hessian-related Fourier components.

        Parameters:
        - v_mu: Mean vector.
        - v_sigma: Covariance matrix (or a vector reshaped into a matrix).
        - N_mc: Number of Monte Carlo samples (default: 1000).
        - alpha: System-level coupling weight.
        - c: Optional inequality-constraint constant.
        - epsilon: Scale used in the domain transformation.
        - fix_damping: If True, reuses damping selected at the first call.
        """
        self._v_mu = np.array(v_mu)
        self._v_sigma = np.array(v_sigma).reshape((len(v_mu), len(v_mu)))
        self._alpha = alpha
        self._epsilon = epsilon
        self._fix_damp = fix_damping
  

        
        self._seed = seed


        self._N_mc= int(N_mc)

        d = len(self._v_mu)

        self._dim = self._v_mu.size


        
        # Common constants for Fourier inversion
        self._coeff_f = -0.5 / np.pi
        self._coeff_g = 1.0 / np.pi
        self._coeff_hl = 1 / (2 * np.pi) ** 2

        self._current_m = None
        
        # ---------------------------------------------------------------------------
        # Last damping parameters and MC diagnostics by component family.
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

        
        self._cache_1d_K = {}
        self._cache_2d_K = {}
        self._cache_2d_val = {}
        self._cache_2d_V_list = {}

        self._mc_points =generate_mc_points(
                    N=self._N_mc, seed=self._seed
                )
    
        self._jac_components = []

        self._jac_cov_history = []
        self._pending_jac_cov = None

        # Cache characteristic-function values for fixed damping parameters.
        self._cf_cache_1d = {}
        self._cf_cache_2d = {}

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
                chol = np.linalg.cholesky(mat)

                self._sigma_trans_2D[(i, j)] = mat
                self._L_2D[(i, j)] = chol

                self._sigma_trans_2D[(j, i)] = mat[np.ix_(swap, swap)]
                self._L_2D[(j, i)] = np.linalg.cholesky(
                    self._sigma_trans_2D[(j, i)]
                )

        # ---------------------------------------------------------------------------
        # Precomputed normal transforms and importance-sampling weights.
        # ---------------------------------------------------------------------------
        lower = np.nextafter(0.0, 1.0)
        upper = np.nextafter(1.0, 0.0)

        points_1d = np.clip(
            self._mc_points[1],
            lower,
            upper,
        )
        points_2d = np.clip(
            self._mc_points[2],
            lower,
            upper,
        )

        self._normal_1d = norm.ppf(points_1d)
        self._normal_2d = norm.ppf(points_2d)

        # 1D transformed samples and reciprocal proposal densities.
        self._transformed_1d = np.empty(
            (
                self._dim,
                self._N_mc,
            ),
            dtype=float,
        )

        self._weights_1d = np.empty_like(
            self._transformed_1d
        )

        for idx in range(self._dim):
            variance_is = float(
                self._sigma_trans_1D[idx]
            )

            if variance_is <= 0.0:
                raise ValueError(
                    "The 1D importance-sampling variance "
                    f"must be positive; got {variance_is} "
                    f"for component {idx}."
                )

            sd_is = np.sqrt(
                variance_is
            )

            self._transformed_1d[idx] = (
                sd_is
                * self._normal_1d
            )

            log_normalization = (
                -0.5 * np.log(2.0 * np.pi)
                - np.log(sd_is)
            )

            self._weights_1d[idx] = np.exp(
                -log_normalization
                + 0.5 * self._normal_1d**2
            )

        # 2D transformed samples and reciprocal proposal densities.
        self._transformed_2d = {}
        self._weights_2d = {}

        quad_2d = np.sum(
            self._normal_2d**2,
            axis=-1,
        )

        for pair, L in self._L_2D.items():
            sigma_is = self._sigma_trans_2D[pair]

            sign, logdet = np.linalg.slogdet(
                sigma_is
            )
            if sign <= 0:
                raise ValueError(
                    "Importance covariance must be positive definite."
                )

            log_normalization = (
                -np.log(2.0 * np.pi)
                - 0.5 * logdet
            )

            self._transformed_2d[pair] = (
                self._normal_2d @ L.T
            )
            self._weights_2d[pair] = np.exp(
                -log_normalization
                + 0.5 * quad_2d
            )
        
        super().__init__(len(v_mu), c)
        


    # ---------------------------------------------------------------------------
    # Domain-transformed covariance used for IS sampling.
    # ---------------------------------------------------------------------------

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

    def _rescale_cov(self,Sigma_block):
        Sigma_block = np.atleast_2d(Sigma_block)
        if Sigma_block.shape == (1, 1):
            sig_scale = self._epsilon * Sigma_block
            sig_tildle = np.linalg.inv(sig_scale)
            return float(sig_tildle[0,0])
        else:
            sig_scale = self._epsilon * Sigma_block
            sig_tildle = np.linalg.inv(sig_scale)
        return sig_tildle

    # ---------------------------------------------------------------------------
    # Adaptive damping selection for component integrands.
    # ---------------------------------------------------------------------------

    def _select_K(self, loss_name,m_curr, idx1=None, idx2=None):

        """Select adaptive damping K for a 1D or 2D component family."""

        if loss_name == "hess_2D_boundary":

            first, elapsed = self._select_K(
                "f", float(m_curr[0]), idx1=idx1
            )
            return np.array([first, 0.0]), elapsed


        # 1D branch (warm start + constrained optimization).
        if idx1 is not None and idx2 is None:
            i = idx1
            sigma_i_sc = self._v_sigma[i,i]
            lam_da = 0  ## Penalizing parameter for boundary-hugging control


            
            def F1_G1(K):
                val_loss,grad_loss,_= self._loss_grad_hess_by_name(loss_name, K, i)
                val_cf,grad_cf,_ = self._log_cf_grad_hess_1d(K,i)
                F1 = val_loss+val_cf + lam_da/2 *(K**2)*sigma_i_sc-K*m_curr
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
        
            K0 = float(res.x.item())

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
        val = -float(scar_vec.dot(np.log(K)))

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

        val = -float(scar_vec.dot(np.log(K)))

        grad = -scar_vec / K
        hess = np.diag(scar_vec / K**2)
        return val,grad, hess
    
    def _loss_hess_1D_grad_hess(self, K):
        val = - np.log(K)
        grad = -1 /K
        hess = 1/ K**2
        return val,grad, hess
    
    def _loss_hess_2D_grad_hess(self, K):
        scar_vec =  np.array([1.0, 1.0], dtype=float)

        val = -float(scar_vec.dot(np.log(K)))

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
    # ============================================================================

    # Characteristic function for component integrands
    def char_function(self, t, i=None, j=None):
        """Characteristic function for full, 1D marginal, or 2D pair inputs."""
        mu = self._v_mu
        Sigma = self._v_sigma
        t_arr = np.asarray(t)

        # Full CF.
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
    def _get_cf_1d(self,idx,K):
        """Return cached 1D characteristic-function values."""
        K = float(K)
        cache_key = (int(idx),K)
        if cache_key not in self._cf_cache_1d:
            U = self._transformed_1d[idx]

            shifted_U = U + 1j * K
            cf_values = self.char_function(
                shifted_U,
                i=idx,
        )
            self._cf_cache_1d[cache_key] = np.asarray(
            cf_values,
            dtype=complex,
        ).reshape(self._N_mc)
        return self._cf_cache_1d[cache_key]

    def g_fourier_integrand_vec(self,u, idx, m_val, K,cf_values = None):
        """Integrand for 1D loss component g."""
        i = 1j
        ui_K = u + i*K
        if cf_values is None:
            cf_values = self.char_function(ui_K, i=idx)
    
        phase = np.exp(i*u * m_val)
        return phase * cf_values / (i * ui_K**3)
        

    def f_fourier_integrand_vec(self,u, idx, m_val, K,cf_values = None):
        """Integrand for 1D gradient component f."""
        i = 1j
        ui_K = u + i*K

        if cf_values is None:
            cf_values = self.char_function(ui_K, i=idx)
        phase = np.exp(i*u * m_val)
        return phase * cf_values / (ui_K**2)
    
    def hess_1D_fourier_integrand_vec(self,u, idx, m_val, K,cf_values = None):
        """Integrand for 1D Hessian helper component."""
        i = 1j
        ui_K = u + i*K
        phase = np.expm1(-(K - 1j * u) * m_val) + 1.0

        if cf_values is None:
                cf_values = self.char_function(ui_K, i=idx)
    
        phase = np.exp(i*u * m_val)
        return phase * cf_values / (i * ui_K)
       

    def _compute_1d(self, loss_name, idx, m_val):
        """
        Compute one 1D component family at index `idx`.

        loss_name in {"g", "f", "hess_1D"}:
        - g: 1D loss component
        - f: 1D gradient component
        - hess_1D: 1D Hessian helper
        """
        key_1D = (loss_name,idx,float(m_val))
        key_damp_1d = (loss_name,idx)
       

        if loss_name == "g":
            coef   = self._coeff_g
     
        elif loss_name == 'f':
            coef  = self._coeff_f
        elif loss_name == "hess_1D":
            coef = self._coeff_f

        if key_1D in self._cache_1d_val:
            return self._cache_1d_val[key_1D],self._cache_1d_V_list[key_1D]

        # Select damping for the current step.
        if self._fix_damp:
            if key_damp_1d in self._cache_1d_K:
                K_1D = self._cache_1d_K[key_damp_1d]
            else:
                K_1D = self._select_K(loss_name, m_val, idx1=idx)
                self._cache_1d_K[key_damp_1d] = K_1D
        else:
            K_1D = self._select_K(loss_name, m_val, idx1=idx)

        U_1D = self._transformed_1d[idx,:]
        importance_weights = self._weights_1d[idx,:]
        cf_values = self._get_cf_1d(idx, K_1D)

        integrand_methods = getattr(self, f"{loss_name}_fourier_integrand_vec")
        integrand_1D = partial(
            integrand_methods,
            cf_values=cf_values,
        )

        est, V_est_1D = MC_Fourier_1D_N_vec(
                K=K_1D,
                idx=idx,
                m_val=m_val,
                integrand_func=integrand_1D,
                U_1D=U_1D,
                importance_weights=importance_weights,
            )


        scale = np.exp(-K_1D * m_val) * coef
        val = scale * est
        V_list_1D = scale * V_est_1D

        if loss_name == "hess_1D":
            return val


        self._last_K1[loss_name][idx]      = K_1D
        self._cache_1d_val[key_1D] = val
        self._cache_1d_V_list[key_1D] = V_list_1D

        return val,V_list_1D
   

    # 2D components
    def _get_cf_2d(self,idx1,idx2,K,transform_key):
        """Return cached 2D characteristic-function values."""

        K = np.asarray(K,dtype = float)
        cache_key = (
            int(idx1),
            int(idx2),
            tuple(transform_key),
            tuple(K.tolist()),
        )
        if cache_key not in self._cf_cache_2d:
            U = np.asarray(
            self._transformed_2d[transform_key],
            dtype=float,
        )

            if U.shape != (self._N_mc, 2):
                raise ValueError(
                    "Expected transformed 2D points with shape "
                    f"({self._N_mc}, 2), but got {U.shape}."
                )

            shifted_U = U + 1j * K[None, :]

            cf_values = self.char_function(
                shifted_U,
                i=idx1,
                j=idx2,
            )

            self._cf_cache_2d[cache_key] = np.asarray(
                cf_values,
                dtype=complex,
            ).reshape(self._N_mc)

        return self._cf_cache_2d[cache_key]

    def h_fourier_integrand_vec(self,u, v, idx1, idx2, x, y, K,cf_values = None):
        """Integrand for 2D loss component h."""
        i = 1j
        ui_K0 = u + i*K[0]
        ui_K1 = v + i*K[1]

        if cf_values is None:
            cf_values = self.char_function(np.stack([ui_K0, ui_K1], axis=-1), idx1, idx2)

        phase = np.exp(i*u*x + i*v*y)
        return phase * cf_values / (ui_K0**2 * ui_K1**2)

    def l_fourier_integrand_vec(self,u, v, idx1, idx2, x, y, K,cf_values = None):
        """Integrand for 2D gradient component l."""
        i = 1j
        ui_K0 = u + i*K[0]
        ui_K1 = v + i*K[1]
        if cf_values is None:
            cf_values = self.char_function(np.stack([ui_K0, ui_K1], axis=-1), idx1, idx2)

        phase = np.exp( i*u*x + i*v*y)
        return phase * cf_values / (i * ui_K0**2 * ui_K1)
    
    def hess_2D_fourier_integrand_vec(self,u,v,idx1,idx2,x,y,K,cf_values = None):
        """Integrand for 2D Hessian helper component."""
        i = 1j
        ui_K0 = u + i*K[0]
        ui_K1 = v + i*K[1]

        if cf_values is None:
            cf_values = self.char_function(np.stack([ui_K0, ui_K1], axis=-1), idx1, idx2)

        phase = np.exp(i*u*x + i*v*y)
        
        return phase * cf_values / (ui_K0 * ui_K1)

    def hess_2D_boundary_fourier_integrand_vec(
        self, u, v, idx1, idx2, x, y, K, cf_values=None
    ):
        '''Intergrand for boundary correction term'''
        ui_K0 = u + 1j * K[0]
        if cf_values is None:
            cf_values = self.char_function(
                np.stack([ui_K0, v], axis=-1), idx1, idx2
            )
        phase = np.exp(1j * (u * x + v * y))
        return phase * cf_values / ui_K0**2


    def _compute_2d(self, loss_name, i, j, m_val):
        """
        Compute one 2D component family for pair (i, j).

        loss_name in {"h", "l", "hess_2D","hess_2D_boundary"}:
        - h: 2D loss component
        - l: 2D gradient component
        - hess_2D: 2D Hessian helper
        - hess_2D_boundary: 2D Hessian for boundary correction term 
        """
        m_val_2D = m_val[[i,j]]
        key = (i,j)
        key_sort = (i,j) if i<j else (j,i)

        
        cache_key = (loss_name, int(i), int(j), tuple(m_val_2D.tolist()))

        key_damp_2d = (loss_name,key)
               
        if cache_key in self._cache_2d_val:
            return self._cache_2d_val[cache_key],self._cache_2d_V_list[cache_key]
        
        
        key_mc = key

        if loss_name == 'hess_2D_boundary':
            indic_K = 'hess_2D'
        else:
            indic_K = loss_name

        # Select damping for the current step.
        if self._fix_damp:
            if key_damp_2d in self._cache_2d_K:
                K_2D= self._cache_2d_K[key_damp_2d]
            else:
                K_2D= self._select_K(loss_name =  indic_K,m_curr = m_val_2D, idx1=i, idx2=j)
                self._cache_2d_K[key_damp_2d] = K_2D
        else:
            K_2D= self._select_K(loss_name =  indic_K, m_curr = m_val_2D, idx1=i, idx2=j)


        U_2D = self._transformed_2d[
            key_mc
        ]
        importance_weights = self._weights_2d[
            key_mc
        ]
        cf_values = self._get_cf_2d(
            idx1=i,
            idx2=j,
            K=K_2D,
            transform_key=key_mc,
        )
        
        integrand_methods = getattr(self, f"{loss_name}_fourier_integrand_vec")
        integrand_2D = partial(
            integrand_methods,
            cf_values=cf_values,
        )

   
        est,V_est_2D = MC_Fourier_2D_N_vec(
                    K=K_2D,
                    idx1 = i,
                    idx2 = j,
                    m_val = m_val_2D,
                    integrand_func=integrand_2D,
                    U_2D = U_2D,
                    importance_weights= importance_weights,
                )
        

        sum_Km = K_2D.T @ m_val_2D
        scale = self._coeff_hl * np.exp(-sum_Km)
        val_2D = scale * est
        V_list_2D = scale * V_est_2D

        if loss_name == "hess_2D":
            return val_2D

        if loss_name == 'hess_2D_boundary':
            Km_first = K_2D[0]*m_val[0]
            val_bound_2d = np.exp(-Km_first) * self._coeff_hl * est
            return val_bound_2d
        

        self._last_K2[loss_name][key_sort]   = K_2D if i<j else K_2D[::-1]
        self._cache_2d_val[cache_key] = val_2D
        self._cache_2d_V_list[cache_key] = V_list_2D
        
        
        return val_2D, V_list_2D
    
    # ============================================================================
    # Aggregation of all Fourier components
    # ============================================================================
    # Loss aggregation: E[l(X - m)] from 1D and 2D components.

    def loss_derivative_samples(self, m):
        """Paired loss/positive-gradient contributions, shapes (N,) and (N,d)."""
        m = self._check_argument(m)
        self._current_m = m.copy()
        d = len(m)

        loss_samples = np.full(self._N_mc, self.e(m), dtype=float)
        gradient_samples = np.ones((self._N_mc, d), dtype=float)

        for i in range(d):
            _, vec_g = self._compute_1d("g", i, m[i])
            _, vec_f = self._compute_1d("f", i, m[i])
            loss_samples += 0.5 * vec_g
            gradient_samples[:, i] += vec_f

            if self._alpha:
                for j in range(d):
                    if j != i:
                        _, vec_l = self._compute_2d("l", j, i, m)
                        gradient_samples[:, i] += self._alpha * vec_l

        if self._alpha:
            for i in range(d):
                for j in range(i + 1, d):
                    _, vec_h = self._compute_2d("h", i, j, m)
                    loss_samples += self._alpha * vec_h

        return loss_samples, gradient_samples


    def shortfall_risk(self, m):
        """Aggregate objective from deterministic term + Fourier loss components."""
        m = self._check_argument(m)

        self._current_m = m.copy()
        d = len(m)

        
        
        e = self.e(m)
        g = sum(self._compute_1d('g', i, m[i])[0] for i in range(d))
        h = 0.0
        if self._alpha:
            h = sum(self._compute_2d('h', i, j,m)[0]
                    for i in range(d) for j in range(i+1, d))
        return e + 0.5*g + (self._alpha * h)
                



    # Loss gradient aggregation: E[∇l(X - m)] from 1D and 2D components.

    def shortfall_risk_jac(self, m):
        """Aggregate gradient using 1D component `f` and 2D component `l`."""
        m = self._check_argument(m)

        
        self._current_m = m.copy()
        d = len(m)


        jac = np.zeros_like(m,dtype = float)
        samples_jac = np.zeros((self._N_mc, d), dtype=float)


        for i in range(d):
            val_f, vec_f = self._compute_1d('f',i,m[i])
            jac[i] = 1+ val_f
            samples_jac[:,i] = 1 + vec_f

            if self._alpha:
                for j in range(d):
                    if j != i:
                        val_l, vec_l = self._compute_2d('l',j,i,m)
                        jac[i] += self._alpha * val_l
                        samples_jac[:,i] += self._alpha * vec_l

        self._pending_jac = samples_jac.copy() 
        self._pending_jac_cov = np.atleast_2d(np.cov(
            samples_jac,
            rowvar=False,
            ddof=1,
        ))
        return jac
        
    # ============================================================================
    # Statistical error components
    # ============================================================================

    def shortfall_risk_jac_var(self, per_iteration=False):
        """
        Return the covariance estimate of Jacobian MC samples.

        If `per_iteration` is True, return the covariance history; otherwise,
        return the latest covariance matrix.
        """
        if not self._jac_cov_history:
            if self._pending_jac_cov is not None:
                if per_iteration:
                    return [self._pending_jac_cov.copy()]
                return self._pending_jac_cov.copy()

            raise RuntimeError(
                "shortfall_risk_jac_var was called before "
                "a Jacobian covariance was computed."
            )

        if per_iteration:
            return self._jac_cov_history

        return self._jac_cov_history[-1]

    # Inverse Hessian component of sandwich covariance V.

    def mean_loss_hessian(self, m):
        """Mean loss Hessian, including the ordered boundary correction."""
        m = self._check_argument(m)
        self._current_m = m.copy()
        d = len(m)
        hessian = np.zeros((d, d), dtype=float)

        for i in range(d):
            value = self._compute_1d("hess_1D", i, m[i])

            hessian[i, i] = value

            if self._alpha:
                for j in range(d):
                    if j == i:
                        continue
                    cross = self._compute_2d("hess_2D", j, i, m)
                    boundary = self._compute_2d("hess_2D_boundary", j, i, m)
                    hessian[i, j] -= self._alpha * cross
                    hessian[i, i] -= self._alpha * boundary

        return hessian


    def gradient_oracle(self, loss_samples, gradient_samples, lam):
        """One full Lagrangian oracle per MC observation: shape (N,d+1)."""
        n_samples, d = gradient_samples.shape
        oracle_samples = np.empty((n_samples, d + 1), dtype=float)
        oracle_samples[:, :d] = 1.0 - lam * gradient_samples
        oracle_samples[:, -1] = loss_samples - self.c
        return oracle_samples


    def jacobian_oracle(self, m, lam, mean_gradient):
        d = len(m)
        

        jacobian = np.zeros((d + 1, d + 1), dtype=float)
        jacobian[:d, :d] = lam * self.mean_loss_hessian(m)
        jacobian[:d, -1] = -mean_gradient
        jacobian[-1, :d] = -mean_gradient
        return jacobian


    def _oracle_sandwich(self, m, lam):
        """Full per-observation sandwich; divide by N for estimator covariance."""
        m = self._check_argument(m)
        loss_samples, gradient_samples = self.loss_derivative_samples(m)
        if gradient_samples.shape[0] < 2:
            raise ValueError("At least two MC observations are required.")

        oracle_samples = self.gradient_oracle(loss_samples, gradient_samples, lam)
        oracle_covariance = np.cov(oracle_samples, rowvar=False, ddof=1)
        mean_jacobian = self.jacobian_oracle(
            m, lam, gradient_samples.mean(axis=0)
        )
        left = np.linalg.solve(mean_jacobian, oracle_covariance)
        return np.linalg.solve(mean_jacobian, left.T).T


    def solution_covariance(self, m, lam):
        """Estimated covariance of (m, lambda), shape (d+1,d+1)."""
        return self._oracle_sandwich(m, lam) / self._N_mc


    def stat_err_full(self, m, lam, alpha_conf=0.05):
        """Marginal confidence half-widths for the allocation components."""
        covariance = self.solution_covariance(m, lam)
        critical_value = norm.ppf(1.0 - alpha_conf / 2.0)
        errors = critical_value * np.sqrt(np.diag(covariance)[:self._dim])
        return float(errors.max())


    def statistical_error_sol_MC(self, m, lam, alpha_conf=0.05):
        """Return the maximum coordinate-wise confidence radius for the solution."""
        return self.stat_err_full(m, lam, alpha_conf=alpha_conf)

# ============================================================================
# End of module
# ============================================================================
