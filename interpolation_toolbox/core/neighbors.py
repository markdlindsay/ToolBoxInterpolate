"""Nearest-neighbour search used by all local interpolators.

Uses scipy's cKDTree when it is installed, otherwise falls back to a chunked
brute-force search written in plain numpy (QGIS always ships numpy, but not
always scipy).
"""

import numpy as np

try:  # pragma: no cover - depends on the user's QGIS install
    from scipy.spatial import cKDTree
except Exception:  # noqa: BLE001
    cKDTree = None

#: Roughly how many float64 values a single brute-force distance block may hold.
_BLOCK_VALUES = 4_000_000


def has_scipy():
    return cKDTree is not None


class Cancelled(Exception):
    """Raised by a progress callback to stop a computation."""


def _report(progress, fraction):
    if progress is not None:
        progress(min(max(fraction, 0.0), 1.0))


class NeighborSearch:
    """k-nearest / fixed-radius neighbour search over 2-D sample points."""

    def __init__(self, xy, use_scipy=True):
        self.xy = np.ascontiguousarray(xy, dtype=float)
        self.n = len(self.xy)
        self.tree = cKDTree(self.xy) if (use_scipy and cKDTree is not None) else None

    def query(self, q, k, radius=None):
        """Return ``(dist, idx)`` arrays of shape ``(len(q), k)``.

        Neighbours are sorted by distance. Slots with no neighbour (because of
        ``radius`` or because ``k`` exceeds the number of points) hold
        ``dist = inf`` and ``idx = -1``.
        """
        q = np.atleast_2d(np.asarray(q, dtype=float))
        m = len(q)
        k_eff = int(min(k, self.n))
        radius = float(radius) if radius and radius > 0 else None

        if self.tree is not None and k_eff < self.n:
            d, i = self.tree.query(
                q, k=k_eff, distance_upper_bound=radius if radius else np.inf
            )
            if k_eff == 1:
                d, i = d[:, None], i[:, None]
            d = np.asarray(d, float)
            i = np.where(np.isfinite(d), i, -1).astype(np.int64)
        elif k_eff < self.n and self.n > 256 and m > 256:
            d, i = self._bucket_query(q, k_eff, radius)
        else:
            d = np.empty((m, k_eff))
            i = np.empty((m, k_eff), dtype=np.int64)
            step = max(1, _BLOCK_VALUES // max(self.n, 1))
            px, py = self.xy[:, 0], self.xy[:, 1]
            for s in range(0, m, step):
                qq = q[s:s + step]
                d2 = (qq[:, 0, None] - px[None, :]) ** 2 + (qq[:, 1, None] - py[None, :]) ** 2
                if k_eff < self.n:
                    idx = np.argpartition(d2, k_eff - 1, axis=1)[:, :k_eff]
                else:
                    idx = np.broadcast_to(np.arange(self.n), d2.shape).copy()
                dd = np.take_along_axis(d2, idx, axis=1)
                order = np.argsort(dd, axis=1)
                d[s:s + step] = np.sqrt(np.take_along_axis(dd, order, axis=1))
                i[s:s + step] = np.take_along_axis(idx, order, axis=1)
            if radius is not None:
                outside = d > radius
                d[outside] = np.inf
                i[outside] = -1

        return self._pad(d, i, k, k_eff, m)

    @staticmethod
    def _pad(d, i, k, k_eff, m):
        if k_eff < k:  # pad so callers always get k columns
            pad = k - k_eff
            d = np.hstack([d, np.full((m, pad), np.inf)])
            i = np.hstack([i, np.full((m, pad), -1, dtype=np.int64)])
        return d, i

    # ------------------------------------------------------------------
    # numpy-only spatial index: points binned into a regular bucket grid
    # ------------------------------------------------------------------
    def _build_buckets(self, k):
        lo = self.xy.min(axis=0)
        span = np.maximum(self.xy.max(axis=0) - lo, 1e-12)
        # aim for ~k/2 points per bucket on average
        n_buckets = max(1.0, 2.0 * self.n / max(k, 1))
        size = float(np.sqrt(span[0] * span[1] / n_buckets)) or float(max(span))
        size = max(size, float(max(span)) / 4096.0)
        nx = int(span[0] // size) + 1
        ny = int(span[1] // size) + 1
        bx = np.minimum(((self.xy[:, 0] - lo[0]) // size).astype(np.int64), nx - 1)
        by = np.minimum(((self.xy[:, 1] - lo[1]) // size).astype(np.int64), ny - 1)
        cell = by * nx + bx
        order = np.argsort(cell, kind="stable")
        starts = np.searchsorted(cell[order], np.arange(nx * ny + 1))
        self._buckets = (k, lo, size, nx, ny, order, starts)

    def _bucket_query(self, q, k, radius):
        if getattr(self, "_buckets", None) is None or self._buckets[0] != k:
            self._build_buckets(k)
        _, lo, size, nx, ny, order, starts = self._buckets
        m = len(q)
        out_d = np.full((m, k), np.inf)
        out_i = np.full((m, k), -1, dtype=np.int64)

        qx = np.clip(((q[:, 0] - lo[0]) // size).astype(np.int64), 0, nx - 1)
        qy = np.clip(((q[:, 1] - lo[1]) // size).astype(np.int64), 0, ny - 1)
        qcell = qy * nx + qx
        qorder = np.argsort(qcell, kind="stable")
        qsorted = qcell[qorder]
        bounds = np.flatnonzero(np.diff(qsorted)) + 1
        groups = np.split(qorder, bounds)
        r_min = int(np.ceil(radius / size)) if radius else 1

        for g in groups:
            cx, cy = int(qx[g[0]]), int(qy[g[0]])
            qq = q[g]
            pending = np.arange(len(g))
            r = 1
            while len(pending):
                x0, x1 = max(cx - r, 0), min(cx + r, nx - 1)
                y0, y1 = max(cy - r, 0), min(cy + r, ny - 1)
                cand = np.concatenate([order[starts[y * nx + x0]:starts[y * nx + x1 + 1]]
                                       for y in range(y0, y1 + 1)])
                full = x0 == 0 and y0 == 0 and x1 == nx - 1 and y1 == ny - 1
                if len(cand) < k and not full and not (radius and r >= r_min):
                    r += 1
                    continue
                p = qq[pending]
                d2 = ((p[:, 0, None] - self.xy[cand, 0][None, :]) ** 2
                      + (p[:, 1, None] - self.xy[cand, 1][None, :]) ** 2)
                kk = min(k, len(cand))
                if kk:
                    idx = (np.argpartition(d2, kk - 1, axis=1)[:, :kk] if kk < len(cand)
                           else np.broadcast_to(np.arange(len(cand)), d2.shape))
                    dd = np.take_along_axis(d2, idx, axis=1)
                    srt = np.argsort(dd, axis=1)
                    dd = np.sqrt(np.take_along_axis(dd, srt, axis=1))
                    ii = cand[np.take_along_axis(idx, srt, axis=1)]
                else:
                    dd = np.full((len(p), 0), np.inf)
                    ii = np.full((len(p), 0), -1, dtype=np.int64)
                # distance from each query to the edge of the searched box;
                # box edges on the grid border are open (no points beyond them)
                inf = np.inf
                left = p[:, 0] - (lo[0] + x0 * size) if x0 > 0 else inf
                right = (lo[0] + (x1 + 1) * size) - p[:, 0] if x1 < nx - 1 else inf
                down = p[:, 1] - (lo[1] + y0 * size) if y0 > 0 else inf
                up = (lo[1] + (y1 + 1) * size) - p[:, 1] if y1 < ny - 1 else inf
                edge = np.minimum(np.minimum(left, right), np.minimum(down, up))
                edge = np.broadcast_to(edge, (len(p),))
                kth = dd[:, kk - 1] if kk == k else np.full(len(p), np.inf)
                done = (kth <= edge) | full
                if radius:
                    done |= edge >= radius
                rows = g[pending[done]]
                out_d[rows, :kk] = dd[done]
                out_i[rows, :kk] = ii[done]
                pending = pending[~done]
                r += 1

        if radius:
            outside = out_d > radius
            out_d[outside] = np.inf
            out_i[outside] = -1
        return out_d, out_i


def chunks(m, k, budget=_BLOCK_VALUES):
    """Yield slices over ``m`` query points so each chunk handles ~budget values."""
    step = max(1, int(budget // max(k, 1)))
    for s in range(0, m, step):
        yield slice(s, min(s + step, m))


def pairwise_distances(a, b):
    """Dense Euclidean distance matrix between two point sets."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.sqrt(
        (a[:, 0, None] - b[None, :, 0]) ** 2 + (a[:, 1, None] - b[None, :, 1]) ** 2
    )
