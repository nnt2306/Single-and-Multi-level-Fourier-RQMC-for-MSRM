import sys
import numpy as np
import time
import qmcpy
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from scipy.linalg import solve_triangular
from scipy.special import kv as bessel_func

path_pjt = "/Users/Nguye071/Documents/GitHub/Single-and-Multi-level-Fourier-RQMC-for-MSRM"
sys.path.insert(0, path_pjt)
from msra_loss import CommonLossFunctionAbs


# Support function to derive RQMC estimate for NIG case
def exponential_pdf(x, sigma_IS):
  """Compute the probability density function of the exponential distribution.

  Args:
  - x (float): Value to evaluate the PDF at.
  - sigma_IS (float): Scale parameter of the exponential distribution.

  Returns:
  - pdf (float): PDF value at x.
  """
  return np.exp(-np.abs(x) / sigma_IS) / (2 * sigma_IS)

def oneD_exponential_cdf(x, sigma):
    x = np.asarray(x, dtype=float)
    lam = 1.0 / sigma
    return np.where(
        x < 0.0,
        0.5 * np.exp(lam * x),
        1.0 - 0.5 * np.exp(-lam * x),
    )
   

def oneD_exponential_inverse_cdf(u, sigma):
    """Compute the component-wise inverse cumulative distribution function of the exponential distribution (multidimensional) of. avector

    Args:
    - u (array): vector to evaluate the inverse CDF at
    - sigma (float): Scale parameter of the exponential distribution.

    Returns:
    - inv_cdf (array): Inverse CDF vector
    """
    u = np.asarray(u, dtype=float)
    # u in (0, 0.5]: negative side
    left  = sigma * np.log(2.0 * u)
    # u in (0.5, 1): positive side
    right = -sigma * np.log(2.0 * (1.0 - u))
    return np.where(u <= 0.5, left, right)
def multivariate_laplace_pdf(x, SIGMA_IS, SIGMA_IS_inv):
  """Compute the PDF of the multivariate Laplace distribution.

  Args:
  - x (array): Value to evaluate the PDF at.
  - SIGMA_IS (array): Covariance matrix of the Laplace distribution.
  - SIGMA_IS_inv (array): Inverse of the covariance matrix.

  Returns:
  - f_ML (float): PDF value.
  """
  d = len(x)
  v = (2 - d) / 2
  f_ML = 2 * (2 * np.pi) ** (-d / 2) * (np.linalg.det(SIGMA_IS)) ** (-0.5) * (x @ SIGMA_IS_inv @ x / 2) ** (v / 2) * bessel_func(v, np.sqrt(2 * x @ SIGMA_IS_inv @ x))
  return f_ML

   
# ============================================================================
# RQMC estimators (NIG IS with domain transformation)
# Notes:
# - `m_prev` is included to evaluate difference-integrands across iterations.
# ============================================================================
def RQMC_Fourier_1D_MNIG_vec(
    K, idx, m_val, m_prev, N, m, sigma_IS, integrand_func, qmc_seq=None
):
    """Compute the 1D RQMC Fourier estimator and per-shift means."""

    sigma_IS = float(np.atleast_1d(sigma_IS).item())

    if qmc_seq is None:
        xi = np.empty((m, N), dtype=float)
        for s in range(m):
            xi[s] = (
                qmcpy.DigitalNetB2(1, graycode=True, randomize="DS", seed=s)
                .gen_samples(N)
                .ravel()
            )
    else:
        xi = np.asarray(qmc_seq, dtype=float).reshape(m, N)

    # Map to Laplace proposal
    U = oneD_exponential_inverse_cdf(xi, sigma_IS)  

    # Integrand values
    F = integrand_func(U, idx, m_val, K, m_prev)  

    log_IS = -np.abs(U) / sigma_IS - np.log(2.0 * sigma_IS)
    W = np.exp(-log_IS)  

    V = np.real(F) * W 
    V_shift_means = V.mean(axis=1)  # per shift
    return V_shift_means.mean(), V_shift_means

# 2D estimator
def RQMC_Fourier_2D_MNIG_vec(
    K, idx1, idx2, m_val, m_prev, N, m, sigma_IS, L_IS, integrand_func, qmc_seq_2D=None
):
    """
    Compute the 2D RQMC Fourier estimator and per-shift means

    Same structure as RQMC_Fourier_2D_MN_vec (Gaussian),
    but Laplace proposal:
        U = r * (L_IS @ Z)       where
            Z ~ N(0, I_2)        (2-D normal from Sobol via inverse CDF)
            r = sqrt(W)          with  W ~ Exp(1)
    pdf_IS(u) = multivariate Laplace pdf evaluated at u.
    """
    m1, m2 = m_val

    sigma_IS = np.asarray(sigma_IS)

    L = np.asarray(L_IS)  # 2×2 lower triangular
    d = 2

    # Sobol points (m, N, 3): first coordinate for Exp(1), next 2 for Gaussian
    if qmc_seq_2D is None:
        xi = np.empty((m, N, 3), dtype=float)
        for s in range(m):
            xi[s] = qmcpy.DigitalNetB2(
                3, graycode=True, randomize="DS", seed=s
            ).gen_samples(N)
    else:
        xi = np.asarray(qmc_seq_2D, dtype=float).reshape(m, N, 3)

    
    u_r = xi[..., 0]  
    r_vals = np.sqrt(-np.log(u_r))  

    
    z_raw = xi[..., 1:]  
    Z = norm.ppf(z_raw) 

    V = Z @ L.T

    U = r_vals[..., None] * V  

    U1 = U[..., 0]
    U2 = U[..., 1]

    F = integrand_func(U1, U2, idx1, idx2, m1, m2, K, m_prev)
    
    U2 = U.reshape(-1, 2)
    W = solve_triangular(L, U2.T, lower=True, overwrite_b=False).T 
    Q = (W**2).sum(axis=1).reshape(m, N)  

    v = (2 - d) / 2.0  

    # Bessel term: K_v(sqrt(2Q))
    s = np.sqrt(2 * Q)
    Kv = bessel_func(v, s)

    # multivariate Laplace pdf:
    # f(u) = 2 (2π)^{-d/2} |Σ|^{-1/2} (Q/2)^{v/2}  K_v( sqrt(2Q) )
    const = 2 * (2 * np.pi) ** (-d / 2) * (np.linalg.det(sigma_IS)) ** (-0.5)
    pdf_vals = const * (Q / 2) ** (v / 2) * Kv  # (m, N)

    W_IS = 1.0 / pdf_vals

    Vvals = np.real(F) * W_IS
    V_shift = Vvals.mean(axis=1)  
    return V_shift.mean(), V_shift

# ============================================================================
# Multi-level / CV Fourier-RQMC class
# ============================================================================
class RQMC_CV_Fou_NIG_qpc(CommonLossFunctionAbs):
    def __init__(
        self,
        v_mu,
        v_sigma,
        v_alpha,
        v_beta,
        v_delta,
        alpha=0,
        c=None,
        N_sobol=64,
        m_shift=30,
        epsilon = 1,
    ):
        """
   
        Initialize the multi-level RQMC Fourier solver for QPC.

        Component naming used throughout this class:
        - g: 1D loss component.
        - f: 1D gradient component (derivative of g contribution).
        - h: 2D loss interaction component.
        - l: 2D gradient interaction component (derivative of h contribution).
        - hess_1D / hess_2D: Hessian-related Fourier components.
        - hess_2D_boundary : Boundary correction term for 2D hessian


        Parameters:
         - v_mu: Location vector.
         - v_beta: Skewness vector.
         - v_alpha: Tail parameter.
         - v_sigma: Covariance matrix.
         - v_delta: Scale (mixing) parameter.

         - alpha: System-level coupling weight.
         - c: Optional constant for the inequality constraint.
         - N_sobol :Number of QMC Sobol point (default value = 64).
         - m_shift: Number of digital net shifts of RQMC (default value = 30).
         - epsilon: Scale used in domain transformation.
        """
        self._v_mu = np.array(v_mu)
        self._v_sigma = np.array(v_sigma).reshape((len(v_mu), len(v_mu)))
        self._alpha = alpha
        self._alpha_nig = v_alpha
        self._epsilon = epsilon


        if self._alpha_nig <= 0:
            raise ValueError("alpha must be positive for MNIG distribution")
        self._beta_nig = np.array(v_beta)
        self._delta_nig = v_delta
        if self._delta_nig <= 0:
            raise ValueError("delta must be positive for MNIG distribution")
        self._N_fixed = int(N_sobol)
        self._N_current_loss= int(N_sobol)
        self._N_current_grad = int(N_sobol)
        self._m_shift = int(m_shift)

        
        # Compute some parameters for the NIG distribution.
        self._beta_sigma = np.dot(self._v_sigma, self._beta_nig)
        self._delta_sig_beta = self._delta_nig * np.dot(self._v_sigma, self._beta_nig)
        self._quad_form = np.dot(self._beta_nig, np.dot(self._v_sigma, self._beta_nig))


        if self._alpha_nig**2 <= self._quad_form:
            raise ValueError(
                "alpha^2 must exceed beta^T Sigma beta for a valid MNIG distribution"
            )
        self._gamma_nig = np.sqrt(self._alpha_nig**2 - self._quad_form)

  
        self._mu_nig = self._v_mu - self._delta_sig_beta / self._gamma_nig


        

        d = len(self._v_mu)


        # Common constants for Fourier inversion
        
        self._coeff_f = -0.5 / np.pi
        self._coeff_g = 0.5 / np.pi
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

        self._var_per_comp = []
        self._cache_1d_K = {}
        self._cache_2d_K = {}
        self._cache_2d_val = {}   # (i, j, tuple(m)) → float
        self._cache_2d_V_list = {}
        self._sobol_seed_1d = 0
        self._sobol_seed_2d = 0

        self._nb_int_2d = 0
        self._nb_int_1d = 0


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
                self._L_2D[(j, i)] = np.linalg.cholesky(
                    self._sigma_trans_2D[(j, i)]
                )

        # ---------------------------------------------------------------------------
        # Set up for difference integrands.
        # ---------------------------------------------------------------------------

        # State used for difference-integrand accumulation across iterations.

        self._prev_m_loss       = None
        self._prev_m_jac = None
    
        self._jac_components = []  
        self._pending_jac = None

        self._cv_history = []
        self._pending_cv_step = None
        self._prev_m_cv = None

        self._accum_loss      = 0.0
        self._accum_jac       = None


        # Memoized objective / gradient
        self._last_loss_m = None
        self._last_loss_val = None
        self._last_jac_m = None
        self._last_jac_val = None

        # ------------------------------------------------------------------

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
        """Reduce sample‐size by 2 each iter, min=4."""
        if grad:
            self._N_current_grad= max(int(self._N_current_grad * mult_factor),4)
        else:
            self._N_current_loss= max(int(self._N_current_loss * mult_factor),4)

    # QMC sequence by dimension (1D/2D) for component integrands.
    def _get_cached_sobol(self, N, m, multi_dim=False, seed_start=None):
        """Return array shape (m, N, dim) using independent base-2 digital shifts."""
        dim = 3 if multi_dim else 1 # in NIG case, we need one more dimension for mixing variable
        cache_sobol = self._sobol_1d_cache if dim ==1 else self._sobol_2d_cache

        if seed_start is None:
            raise ValueError("seed_start must be provided so we can keep the sequences independent.")
        key = (int(N), int(m), dim,int(seed_start))

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
        Sigma_block = np.atleast_2d(Sigma_block)
        if Sigma_block.shape == (1, 1):
            sig_scale = self._epsilon / self._delta_nig
            return sig_scale
        else:
            sig_scale = (self._delta_nig**2 * Sigma_block) / (2 * self._epsilon)
            sig_tildle = np.linalg.inv(sig_scale)
        return sig_tildle


    def _loss_grad_hess_by_name(self, name, K, idx1=None, idx2=None):  #

        """
        Return (grad, hess) both as full‐d arrays/matrices.
        - For 'g' or 'f', it's just the diagonal 1D case.
        - For 'h' or 'l', idx1,idx2 must be provided; we embed the 2×2 block.
        """
        d = len(self._v_mu)

        if name in ("g", "f","hess_1D"):
            if idx1 is None:
                raise ValueError(f"idx1 must be provided for loss '{name}'")
            val_1d,grad_1d, hess_1d = getattr(self, f"_loss_{name}_grad_hess")(K)
            return val_1d,grad_1d, hess_1d

        # 2D terms:
        if name in ("h", "l","hess_2D"):
            # slice out the 2‐vector
            if idx1 is None or idx2 is None:
                raise ValueError(f"idx1 and idx2 must be provided for loss '{name}'")
            i,j = (idx1,idx2) if idx1<idx2 else (idx2,idx1)
            
            val2,grad2, hess2 = getattr(self, f"_loss_{name}_grad_hess")(K)
            
            return val2,grad2, hess2


        raise ValueError(f"Unknown loss name {name!r}")
    # ---------------------------------------------------------------------------
    # Adaptive optimal damping selection for component integrands
    # ---------------------------------------------------------------------------

    def _select_K(self, loss_name, m_curr, idx1=None, idx2=None, m_prev = None):
        """Select adaptive damping K for a 1D or 2D difference component family."""

        if m_prev is not None:
            m_curr = m_curr-m_prev
        ## Penalizing parameter for boundary-hugging control
 
        lam_da = (200 * self._delta_nig)/self._gamma_nig # This approximate 0.2
        
        # 1D branch
        if idx1 is not None and idx2 is None:
            i = idx1
            sigma_i_sc = self._v_sigma[i,i]
            
            def F1_G1(K):
                val_loss,grad_loss,_= self._loss_grad_hess_by_name(loss_name, K, i)
                val_cf,grad_cf,_,_ = self._log_cf_grad_hess_1d(K,i)
                F1 = val_loss+val_cf + lam_da/2 *(K**2)*sigma_i_sc-K*m_curr
                G1 = (grad_loss+grad_cf)  + lam_da*K*sigma_i_sc-m_curr
                return  F1,G1
            
            eps = 0.01

            if self._last_K1[loss_name][i] is None:

                K0 = 1
               
                self._last_K1[loss_name][i] = float(K0)

            def NIG_cons2(K):
                _, _, _, DK = self._log_cf_grad_hess_1d(K, idx=i)
                return DK - eps

            cons_K = {"type": "ineq", "fun": NIG_cons2}

            
            x0 = self._last_K1[loss_name][i]

            bnds = Bounds([eps],[np.inf], keep_feasible=True)

            res = minimize(
                fun= lambda K:F1_G1(K)[0],
                x0=x0,
                method="SLSQP",
                jac=lambda K:F1_G1(K)[1],
                bounds = bnds,
                constraints=cons_K
            )
            K0 = float(res.x.item())

            if NIG_cons2(K0) >= 0 and K0 >0:
                pass 
            else:
                K0 = x0
                self._last_K1[loss_name][i] = K0

            return K0

        # 2D branch (pairwise component).

        if idx1 is not None and idx2 is not None:
            i, j = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
            sigma_2d = self._v_sigma[np.ix_([i, j], [i, j])]
                        
            def F2_G2(K_vec):
                val_loss,grad_loss,_ = self._loss_grad_hess_by_name(loss_name, K_vec, idx1=i,idx2=j)
                val_cf,grad_cf,_,_ = self._log_cf_grad_hess_2d(K_vec,idx1= i,  idx2 = j)
                F2 = val_loss+val_cf + lam_da/2 * float(np.dot(K_vec, sigma_2d @ K_vec))-K_vec.T.dot(m_curr)
                G2 = grad_loss+ grad_cf + lam_da*sigma_2d @ K_vec-m_curr
                return  F2,G2
            
            
            if self._last_K2[loss_name][(i, j)] is None:

                K0_vec = np.ones(2)
                self._last_K2[loss_name][(i, j)] = K0_vec.copy()
            
            eps = 0.01

            def NIG_cons2(K):
                # NIG constraint: alpha^2 > beta^T Sigma beta
                _, _, _, D_pair = self._log_cf_grad_hess_2d(K, idx1=i, idx2=j)
                return D_pair - eps
            
            cons_K = [{"type": "ineq", "fun": NIG_cons2}]


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
            # 8) Feasibility check
            if (np.all(K0_vec > 0)) and (NIG_cons2(K0_vec) >= 0):
                pass 
            else:
                K0_vec = x0_vec


            return K0_vec


    # ---------------------------------------------------------------------------
    # Log-transformed loss and CF terms used in damping optimization.
    # ---------------------------------------------------------------------------

    def _log_cf_grad_hess_1d(self, K_scalar, idx):
       
        mu_i = self._mu_nig[idx]
        sigma_ii = self._v_sigma[idx, idx]
        sigma_beta_i = self._beta_sigma[idx]
        beta_iK = self._beta_nig[idx] - K_scalar

       
        D_K = (
            self._gamma_nig**2 + 2 * K_scalar * sigma_beta_i - (K_scalar**2) * sigma_ii
        )
        num_K = self._delta_nig * sigma_ii * beta_iK

        val = -K_scalar * mu_i + num_K * (self._gamma_nig - np.sqrt(D_K))
        grad = -mu_i - num_K / np.sqrt(D_K)
        hess = self._delta_nig * (
            sigma_ii / np.sqrt(D_K) + (beta_iK**2 * sigma_ii**2) / np.sqrt(D_K**3)
        )
        return val, grad, hess, D_K

    def _log_cf_grad_hess_2d(self, K_pair, idx1, idx2):
        mu_sub = self._mu_nig[[idx1, idx2]]
        sigma_beta_sub = self._beta_sigma[[idx1, idx2]]
        beta_sub_iK = self._beta_nig[[idx1, idx2]] - K_pair
        beta_re = beta_sub_iK.reshape(-1, 1)
        sigma_sub = self._v_sigma[np.ix_([idx1, idx2], [idx1, idx2])]

        D_pair = (
            self._gamma_nig**2
            + 2 * np.dot(K_pair, sigma_beta_sub)
            - K_pair.T @ sigma_sub @ K_pair
        )

        num_pair = self._delta_nig * sigma_sub.dot(beta_sub_iK)
        term_2 = sigma_sub @ beta_re @ beta_re.T @ sigma_sub
        val = -K_pair.T.dot(mu_sub) + self._delta_nig * (
            self._gamma_nig - np.sqrt(D_pair)
        )
        grad = -mu_sub - num_pair / np.sqrt(D_pair)
        hess = self._delta_nig * (sigma_sub / np.sqrt(D_pair) + term_2 / (D_pair**1.5))
        return val, grad, hess, D_pair

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

        val = -float(scar_vec.dot(np.log(K)))  

        grad = -scar_vec / K
        hess = np.diag(scar_vec / K**2)
        return val,grad, hess
    
    def _maybe_scalar(self, arr):
        arr = np.asarray(arr)
        return arr.item() if arr.shape == () else arr
    
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

    # Characteristic function for component integrands
    def char_function_vec(self, t, i=None, j=None):
    
        """Characteristic function for full, 1D marginal, or 2D pair inputs."""

        mu = self._mu_nig
        Sigma = self._v_sigma

        # --- Full CF ---
        if i is None and j is None:
            
            t_arr = np.asarray(t)
            squeezed = False
            if t_arr.ndim == 0:
                t_arr = t_arr.reshape(1, 1)
                squeezed = True
            elif t_arr.ndim == 1:
                t_arr = t_arr[None, ...]
                squeezed = True
            if t_arr.shape[-1] != mu.shape[0]:
                raise ValueError(
                    f"t has last dim {t_arr.shape[-1]} but mu has dim {mu.shape[0]}"
                )
            lin = 1j * np.einsum("...d,d->...", t_arr, mu, optimize=True) 
            beta_plus_iu = self._beta_nig + 1j * t_arr
            quad = np.einsum(
                "...d,df,...f->...",
                beta_plus_iu,
                Sigma,
                beta_plus_iu,
                optimize=True,
            )
            D = self._alpha_nig**2 - quad
            out = np.exp(lin + self._delta_nig * (self._gamma_nig - np.sqrt(D)))
            return out[0] if squeezed else out

        # 1D marginal CF
        elif i is not None and j is None:
            ti = np.asarray(t)
            mu_i = mu[i]
            sigma_beta_i = self._beta_sigma[i]
            sigma_ii = Sigma[i, i]
            
            second_term = 2 * 1j * ti * sigma_beta_i
            third_term = sigma_ii * ti**2 
            D_iu = self._gamma_nig**2 - second_term + third_term
            out = np.exp(
                1j * mu_i * ti + self._delta_nig * (self._gamma_nig - np.sqrt(D_iu))
            )
            return self._maybe_scalar(out)
        
        # 2D pairwise CF
        elif i is not None and j is not None:
            tij = np.asarray(t)
            squeezed = False
            if tij.ndim == 1:
                tij = tij[None, ...]
                squeezed = True

            if tij.shape[-1] != 2:
                raise ValueError("For pairwise CF, t must have last dimension 2.")
            if i > j:
                i, j = j, i
                tij = tij[..., ::-1]


            t1 = tij[..., 0]
            t2 = tij[..., 1]            # extract 2D parameters
            lin = 1j * (mu[i] * t1 + mu[j] * t2)

            sigma_beta_sub = self._beta_sigma[[i, j]]
            Sigma_sub = Sigma[np.ix_([i, j], [i, j])]
            

            # build 2D CF
            second_term = 2 * 1j * (
                sigma_beta_sub[0] * t1 + sigma_beta_sub[1] * t2
            )
            third_term = np.einsum(
                "...d,df,...f->...",
                tij,
                Sigma_sub,
                tij,
                optimize=True,
            )

            D_iu = self._gamma_nig**2 - second_term + third_term
            out = np.exp(
                lin + self._delta_nig * (self._gamma_nig - np.sqrt(D_iu))
            )
            return out[0] if squeezed else out

        else:
            raise RuntimeError("Invalid combination of i, j")
    

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
        # = -K + i*u
        phase = self._phase_1d(u, m_val, K, m_prev)  # stable when delta -> 0
        cf = self.char_function_vec(ui_K, idx)
        return phase * cf / (1j * ui_K**3)



    def f_fourier_integrand_vec(self, u, idx, m_val, K, m_prev=None):
        """Integrand for 1D loss gradient component f."""

        ui_K = u + 1j * K
        # = -K + i*u
        phase = self._phase_1d(u, m_val, K, m_prev)  # stable when delta -> 0
        cf = self.char_function_vec(ui_K, idx)
        return phase * cf / ui_K**2


    def hess_1D_fourier_integrand_vec(
            self, u, idx, m_val, K, m_prev=None):

        """Integrand for 1D Hessian component."""

        ui_K = u + 1j * K
        phase = self._phase_1d(u, m_val, K, m_prev)
        cf = self.char_function_vec(ui_K, idx)
        return phase * cf / (1j * ui_K)


    def _compute_1d(self, loss_name, idx, m_val, m_prev = None):
        """
        Compute one 1D component family at index `idx`.

        loss_name in {"g", "f", "hess_1D"}:
        - g: 1D loss component
        - f: 1D gradient component
        - hess_1D: 1D Hessian helper
        If `m_prev` is provided, compute the difference-integrand increment.
        """
        # pick integrand & coefficient
        key_1D = (loss_name,idx,float(m_val))

        key_damp_1d = (loss_name,idx)

        if loss_name == "g":
            coef   = self._coeff_g
            N_current = self._N_current_loss
        elif loss_name == 'f':
            coef  = self._coeff_f
            N_current = self._N_current_grad
        elif loss_name == "hess_1D":
            coef = self._coeff_f
            N_current = self._N_fixed

        
        integrand = getattr(self, f"{loss_name}_fourier_integrand_vec")
     
        key_parts = [loss_name, idx, float(m_val)]
        
        if m_prev is not None:
            key_parts.append(float(m_prev))

        key_1D = tuple(key_parts)
        
        if key_1D in self._cache_1d_val:
            return self._cache_1d_val[key_1D],self._cache_1d_V_list[key_1D]
        
        # Select damping for current step.

        K_new= self._select_K(loss_name, m_val, idx1=idx, m_prev = m_prev)

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
            seed_start= seed_start
        )
        



        est, V_list_1D = RQMC_Fourier_1D_MNIG_vec(
            K_new, idx,
            m_val,
            m_prev,
            N_current, self._m_shift,
            sigma_IS=self._sigma_trans_1D[idx],
            integrand_func=integrand,
            qmc_seq= qmc_seq_1D
        )


        anchor = m_val if m_prev is None else m_prev
        sum_Km = K_new * float(np.asarray(anchor).item())

        scale = np.exp(-sum_Km) * coef

        val_1D = scale * est
        V_list_1D = scale * V_list_1D


        if loss_name == "hess_1D":
            return val_1D

        self._last_K1[loss_name][idx]      = K_new
        

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
        cf = self.char_function_vec(
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
        cf = self.char_function_vec(
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
        cf = self.char_function_vec(
            np.stack([ui_K0, ui_K1], axis=-1), idx1, idx2
        )

        return phase * cf / (ui_K0 * ui_K1)

    

    def hess_2D_boundary_fourier_integrand_vec(
        self, u, v, idx1, idx2, x, y, K, xy_prev=None
    ):
        """Intergrand for the boundary correction term"""
        boundary_K = (K[0], 0.0)
        ui_K0 = u + 1j * K[0]

        phase = self._phase_2d(
            u, v, x, y, boundary_K, xy_prev
        )
        cf = self.char_function_vec(
            np.stack([ui_K0, v], axis=-1), idx1, idx2
        )

        return phase * cf / ui_K0**2


    def _compute_2d(self, loss_name, i, j, m_val,m_prev = None):
        """
        Compute one 2D component family for pair (i, j).

        loss_name in {"h", "l", "hess_2D"}:
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
        key_parts = [loss_name,key,tuple(m_val)]

        

        key_2D = (loss_name,key,tuple(m_val_2D))
        
        key_damp_2d = (loss_name,key)

        integrand = getattr(self, f"{loss_name}_fourier_integrand_vec")
        if m_prev is not None:
            m_prev_2D = m_prev[[i,j]]
            key_parts.append(tuple(m_prev_2D))
        else:
            m_prev_2D = None

        key_2D = tuple(key_parts)
        
        coef     = self._coeff_hl
        
        
        if key_2D in self._cache_2d_val:
            return self._cache_2d_val[key_2D],self._cache_2d_V_list[key_2D]
        
        
        key_qmc = key_sort

        if loss_name == 'hess_2D_boundary':
            indic_K = 'hess_2D'
        else:
            indic_K = loss_name

        # Select damping for current step.

        K_new = self._select_K(loss_name = indic_K, m_curr = m_val_2D, idx1=i, idx2=j,m_prev = m_prev_2D)

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
            seed_start=seed_start
        )
                
        


        est, V_list_2D = RQMC_Fourier_2D_MNIG_vec(
            K_new, i, j,
            m_val_2D,
            m_prev_2D,
            N_current, self._m_shift,
            sigma_IS= self._sigma_trans_2D[key_qmc],
            L_IS = self._L_2D[key_qmc],
            integrand_func=integrand,
            qmc_seq_2D= qmc_seq_2D
        )
        
        self._nb_int_2d +=1

        anchor = m_val_2D if m_prev_2D is None else m_prev_2D

        if loss_name == "hess_2D_boundary":

            sum_Km = K_new[0] * anchor[0]
        else:
            sum_Km = K_new.T @ anchor

        scale = np.exp(-sum_Km) * coef
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

        # 1D terms
        for i in range(d):
            prev_i = None if m_prev is None else m_prev[i]

            # must use CRN and the SAME pair-level K inside _compute_1d when m_prev is given
            _, vec_g = self._compute_1d(
                "g", i, m[i], m_prev=prev_i
            )
            samples += 0.5 * vec_g

        # 2D terms
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
        """Aggregate loss gradient using baseline or CV-difference updates."""

        _, gradient_value = self._evaluate_cv_step(m, commit=commit)
        return gradient_value

    # ============================================================================
    # Statistical error components
    # ============================================================================
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
            oracle_samples[:, -1] -= self.c

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
                    hessian[i, j] = -self._alpha * val_cross


                    val_boundary = self._compute_2d(
                        'hess_2D_boundary', j, i, m
                    )
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

        return  float(stat_errs.max())

# ============================================================================
