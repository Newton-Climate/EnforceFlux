"""Core matrix utilities and primary information-metric computations."""
import numpy as np

from enforceflux.analysis.information_models import DofResult, FisherResult, PosteriorResult


# ── Fast Woodbury path for spatial inversions (diagonal Sa/Se, n >> m) ────────

def _woodbury_sx_diag(
    G: np.ndarray,   # (m, n)
    Se: np.ndarray,  # (m,) diagonal
    Sa: np.ndarray,  # (n,) diagonal
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal of Sx and the quadratic form vector via Woodbury identity.

    Woodbury: Sx = Sa - Sa G^T (Se + G Sa G^T)^{-1} G Sa

    Returns (sx_diag, quad) both of shape (n,).
    quad[j] = G[:, j]^T M^{-1} G[:, j]  (per-cell sensitivity)
    """
    if mask is not None:
        G = G[mask]
        Se = Se[mask]

    GSa = G * Sa[None, :]              # (m, n)
    M = np.diag(Se) + GSa @ G.T       # (m, m) — small, trivially invertible
    M_inv = np.linalg.inv(M)
    MiG = M_inv @ G                    # (m, n)
    quad = (G * MiG).sum(axis=0)      # (n,)  quad[j] = G[:, j]^T M^{-1} G[:, j]
    sx_diag = Sa * (1.0 - Sa * quad)  # exact diag of Sx = Sa - Sa G^T M^{-1} G Sa
    return sx_diag, quad


def analyze_information_content_spatial(
    G: np.ndarray,   # (m, n) with n >> m
    Se: np.ndarray,  # (m,) diagonal
    Sa: np.ndarray,  # (n,) diagonal, one prior variance per source cell
    obs_groups: dict[str, np.ndarray] | None = None,
    source_names: list[str] | None = None,
) -> tuple[FisherResult, DofResult, PosteriorResult]:
    """O(m²·n) information analysis exploiting diagonal Sa and Se.

    Uses the Woodbury matrix identity to avoid constructing the (n×n) FIM.
    Eigenvalues are derived from the (m×m) dual-space matrix.
    The averaging_kernel field returns the diagonal of A as a (n,) 1D array
    rather than the full (n×n) matrix (which would require n²*8 bytes of RAM).
    """
    G = np.asarray(G, dtype=float)
    Se = np.asarray(Se, dtype=float)
    Sa = np.asarray(Sa, dtype=float)
    m, n = G.shape

    if obs_groups is None:
        obs_groups = {"all": np.ones(m, dtype=bool)}

    se_inv = 1.0 / (Se + 1e-300)

    # Active observation set = union of all groups. This lets callers analyse a
    # subset of the network (e.g. an ablation) by passing only the relevant
    # groups; the joint posterior then reflects exactly those observations.
    active_mask = np.zeros(m, dtype=bool)
    for mask in obs_groups.values():
        active_mask |= np.asarray(mask, dtype=bool)

    # Per-cell diagonal Fisher information (diagnostic only — NOT the DFS),
    # over the active observations; never sum across (possibly nested) groups,
    # which double-counts shared measurements and inflates the Fisher info.
    fim_total_per_cell = (G[active_mask] ** 2 * se_inv[active_mask][:, None]).sum(axis=0)
    fim_per_cell_groups: dict[str, np.ndarray] = {}
    for name, mask in obs_groups.items():
        mask = np.asarray(mask, dtype=bool)
        fim_per_cell_groups[name] = (G[mask] ** 2 * se_inv[mask][:, None]).sum(axis=0)

    # Exact joint posterior diagonal via Woodbury over the active observations.
    # The averaging-kernel trace Σ_j (1 − Sx_jj/Sa_j) = Tr[A] = Σ λ/(1+λ) is
    # bounded by the number of active observations — a per-cell diagonal-FIM
    # sum is NOT (it can exceed the observation count).
    sx_diag, _ = _woodbury_sx_diag(G, Se, Sa, mask=active_mask)  # (n,) exact diag of Sx
    sx_diag = np.clip(sx_diag, 0.0, Sa)              # guard fp round-off
    dfs_per_cell = 1.0 - sx_diag / (Sa + 1e-300)     # (n,) exact A diagonal
    ur = 1.0 - np.sqrt(sx_diag / (Sa + 1e-300))      # (n,)
    post_sigma = np.sqrt(sx_diag)                    # (n,)
    prior_sigma = np.sqrt(Sa)                        # (n,)

    dfs_total = float(dfs_per_cell.sum())            # Tr[A] ≤ m (n_obs)

    # Per-group DFS and posterior — each via exact Woodbury on that obs subset.
    dfs_per_group: dict[str, float] = {}
    Sx_per_group: dict[str, np.ndarray] = {}
    for name, mask in obs_groups.items():
        mask = np.asarray(mask, dtype=bool)
        sx_k, _ = _woodbury_sx_diag(G, Se, Sa, mask=mask)
        sx_k = np.clip(sx_k, 0.0, Sa)
        dfs_per_group[name] = float((1.0 - sx_k / (Sa + 1e-300)).sum())
        Sx_per_group[name] = sx_k

    # Eigenvalues of FIM from the (m×m) dual-space: Se_inv^{1/2} G Sa G^T Se_inv^{1/2}
    # These are the m nonzero eigenvalues of the rank-m FIM (all others are 0).
    G_scaled = G * np.sqrt(se_inv)[:, None]  # Se^{-1/2} G
    M_small = G_scaled @ (Sa[:, None] * G_scaled.T)  # (m, m)
    evals_small, evecs_small = np.linalg.eigh(M_small)
    order = np.argsort(evals_small)[::-1]
    evals_small, evecs_small = evals_small[order], evecs_small[:, order]

    # Build FisherResult with a per-cell 1D "FIM" (not the n×n matrix)
    # FIM_per_group stores scalar arrays (n,); FIM_total stores (n,)
    fisher = FisherResult(
        FIM_total=fim_total_per_cell,          # (n,) — diagonal only
        FIM_per_group=fim_per_cell_groups,
        eigenvalues=evals_small,               # (m,) nonzero eigenvalues
        eigenvectors=evecs_small,
        group_names=list(obs_groups.keys()),
        n_obs_per_group={k: int(np.asarray(v).sum()) for k, v in obs_groups.items()},
    )

    # averaging_kernel: store 1D diagonal of A (A_jj = dfs_per_cell[j])
    dof = DofResult(
        dfs_total=dfs_total,
        dfs_per_group=dfs_per_group,
        dfs_per_source=dfs_per_cell,
        averaging_kernel=dfs_per_cell,         # 1D diagonal — NOT the full n×n matrix
        averaging_kernel_per_group={
            k: 1.0 - Sx_per_group[k] / (Sa + 1e-300)
            for k in obs_groups
        },
        source_names=source_names,
    )

    posterior = PosteriorResult(
        Sx=sx_diag,                            # 1D diagonal
        posterior_sigma=post_sigma,
        prior_sigma=prior_sigma,
        uncertainty_reduction=ur,
        correlation_matrix=np.eye(1),          # diagonal Sa → no cross-correlation
        Sx_per_group=Sx_per_group,
        source_names=source_names,
    )

    return fisher, dof, posterior


def _to_sa_inv(Sa: np.ndarray) -> np.ndarray:
    """Return Sa^{-1} as a full matrix regardless of whether Sa is 1-D or 2-D."""
    Sa = np.asarray(Sa, dtype=float)
    if Sa.ndim == 1:
        return np.diag(1.0 / (Sa + 1e-300))
    return np.linalg.inv(Sa)


def _prior_sigma(Sa: np.ndarray) -> np.ndarray:
    """Extract per-source prior sigma from diagonal or full Sa."""
    Sa = np.asarray(Sa, dtype=float)
    if Sa.ndim == 1:
        return np.sqrt(Sa)
    return np.sqrt(np.diag(Sa))


def _invert_posterior(FIM: np.ndarray, Sa_inv: np.ndarray) -> np.ndarray:
    """Compute Sx = (FIM + Sa^{-1})^{-1} stably via Cholesky when possible."""
    M = FIM + Sa_inv
    try:
        L = np.linalg.cholesky(M + 1e-12 * np.eye(M.shape[0]))
        L_inv = np.linalg.inv(L)
        return L_inv.T @ L_inv
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def _group_fim(G: np.ndarray, Se: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """FIM contribution from a single observation group."""
    mask = np.asarray(mask, dtype=bool)
    G_k = G[mask]
    if Se.ndim == 1:
        se_k_inv = 1.0 / (Se[mask] + 1e-300)
        return (G_k * se_k_inv[:, None]).T @ G_k
    Se_k = Se[np.ix_(mask, mask)]
    return G_k.T @ np.linalg.inv(Se_k) @ G_k


def compute_fisher(
    G: np.ndarray,
    Se: np.ndarray,
    obs_groups: dict[str, np.ndarray] | None = None,
) -> FisherResult:
    """Compute FIM = G^T Se^{-1} G, decomposed by observation group."""
    G = np.asarray(G, dtype=float)
    Se = np.asarray(Se, dtype=float)
    m, _n = G.shape

    if obs_groups is None:
        obs_groups = {"all": np.ones(m, dtype=bool)}

    FIM_per_group: dict[str, np.ndarray] = {}
    n_obs_per_group: dict[str, int] = {}
    for name, mask in obs_groups.items():
        mask = np.asarray(mask, dtype=bool)
        FIM_per_group[name] = _group_fim(G, Se, mask)
        n_obs_per_group[name] = int(mask.sum())

    FIM_total: np.ndarray = sum(FIM_per_group.values())  # type: ignore[assignment]

    evals, evecs = np.linalg.eigh(FIM_total)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    return FisherResult(
        FIM_total=FIM_total,
        FIM_per_group=FIM_per_group,
        eigenvalues=evals,
        eigenvectors=evecs,
        group_names=list(obs_groups.keys()),
        n_obs_per_group=n_obs_per_group,
    )


def compute_dof(
    fisher: FisherResult,
    Sa: np.ndarray,
    source_names: list[str] | None = None,
) -> DofResult:
    """Compute averaging kernel A and degrees of freedom for signal (DFS)."""
    Sa_inv = _to_sa_inv(Sa)
    Sx = _invert_posterior(fisher.FIM_total, Sa_inv)
    A = Sx @ fisher.FIM_total

    dfs_total = float(np.trace(A))
    dfs_per_source = np.diag(A)

    A_per_group: dict[str, np.ndarray] = {}
    dfs_per_group: dict[str, float] = {}
    for name, FIM_k in fisher.FIM_per_group.items():
        A_k = Sx @ FIM_k
        A_per_group[name] = A_k
        dfs_per_group[name] = float(np.trace(A_k))

    return DofResult(
        dfs_total=dfs_total,
        dfs_per_group=dfs_per_group,
        dfs_per_source=dfs_per_source,
        averaging_kernel=A,
        averaging_kernel_per_group=A_per_group,
        source_names=source_names,
    )


def compute_posterior(
    fisher: FisherResult,
    Sa: np.ndarray,
    source_names: list[str] | None = None,
) -> PosteriorResult:
    """Compute Gaussian posterior covariance Sx = (FIM + Sa^{-1})^{-1}."""
    Sa_inv = _to_sa_inv(Sa)
    Sx = _invert_posterior(fisher.FIM_total, Sa_inv)

    post_sigma = np.sqrt(np.diag(Sx))
    prior_sig = _prior_sigma(Sa)
    ur = 1.0 - post_sigma / (prior_sig + 1e-300)

    D = np.diag(1.0 / (post_sigma + 1e-300))
    corr = D @ Sx @ D

    Sx_per_group: dict[str, np.ndarray] = {}
    for name, FIM_k in fisher.FIM_per_group.items():
        Sx_per_group[name] = _invert_posterior(FIM_k, Sa_inv)

    return PosteriorResult(
        Sx=Sx,
        posterior_sigma=post_sigma,
        prior_sigma=prior_sig,
        uncertainty_reduction=ur,
        correlation_matrix=corr,
        Sx_per_group=Sx_per_group,
        source_names=source_names,
    )


def analyze_information_content(
    G: np.ndarray,
    Se: np.ndarray,
    Sa: np.ndarray,
    obs_groups: dict[str, np.ndarray] | None = None,
    source_names: list[str] | None = None,
) -> tuple[FisherResult, DofResult, PosteriorResult]:
    """One-shot information-content analysis."""
    fisher = compute_fisher(G, Se, obs_groups=obs_groups)
    dof = compute_dof(fisher, Sa, source_names=source_names)
    posterior = compute_posterior(fisher, Sa, source_names=source_names)
    return fisher, dof, posterior
