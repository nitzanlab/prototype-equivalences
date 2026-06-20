import numpy as np
import ot


def ot_velocities(X: np.ndarray, timepoints: np.ndarray, sinkhorn: bool = False, reg: float = 0.1,
                  extend_last: bool = False, k: int = 10):
    velocities = np.zeros_like(X)
    timepoints = np.asarray(timepoints, dtype=float)
    times = np.sort(np.unique(timepoints))

    for i in range(len(times)-1):
        src = timepoints == times[i]
        tgt = timepoints == times[i+1]
        dt = times[i+1] - times[i]
        X_src = X[src]
        X_tgt = X[tgt]

        # uniform marginals
        a = np.ones(len(X_src)) / len(X_src)
        b = np.ones(len(X_tgt)) / len(X_tgt)

        # cost matrix: squared euclidean distance, normalized for numerical stability
        M = ot.dist(X_src, X_tgt)
        M = M / M.max()

        if sinkhorn:
            T = ot.sinkhorn(a, b, M, reg=reg)
        else:
            T = ot.emd(a, b, M)

        # velocity: for each source, a weighted average displacement toward the targets
        T_norm = T / T.sum(axis=1, keepdims=True)
        vels = T_norm @ X_tgt - X_src   # shape (src, dim)
        velocities[src] = vels/dt

    if extend_last:
        last_mask = timepoints == times[-1]
        X_last = X[last_mask]
        dists = np.sum((X_last[:, None] - X_last[None, :])**2, axis=-1)
        np.fill_diagonal(dists, np.inf)
        nn_idx = np.argsort(dists, axis=1)[:, :k]
        velocities[last_mask] = X_last[nn_idx].mean(axis=1) - X_last

    return velocities