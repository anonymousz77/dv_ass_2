"""Vectorised Marching Squares isoline extraction (used by the contour-mapping task).

Grid convention: F has shape (ny, nx) with coordinates y (rows) and x (columns).
A cell spans rows i..i+1 and columns j..j+1 with corners
    a = F[i, j]   b = F[i, j+1]
    d = F[i+1, j] c = F[i+1, j+1]
Case index = 8*[a>=iso] + 4*[b>=iso] + 2*[c>=iso] + 1*[d>=iso]  (16 cases).
Edges: 0 = a-b (top), 1 = b-c (right), 2 = c-d (bottom), 3 = d-a (left).
Saddle cases 5 and 10 are resolved with the cell-centre value (mean of the four corners).
Cells with a missing (NaN) corner are skipped, which is how the ocean / outside-CONUS mask is handled.
"""
import numpy as np

# case -> list of (edge, edge) segments; saddles handled separately
CASE_SEGMENTS = {
    0: [], 1: [(3, 2)], 2: [(2, 1)], 3: [(3, 1)], 4: [(0, 1)], 6: [(0, 2)], 7: [(0, 3)],
    8: [(0, 3)], 9: [(0, 2)], 11: [(0, 1)], 12: [(3, 1)], 13: [(1, 2)], 14: [(3, 2)], 15: [],
}
SADDLE_SEGMENTS = {  # (case, centre_above) -> segments
    (5, True): [(0, 3), (1, 2)], (5, False): [(0, 1), (3, 2)],
    (10, True): [(0, 1), (3, 2)], (10, False): [(0, 3), (1, 2)],
}


def _edge_points(edge, a, b, c, d, iso, xj, xj1, yi, yi1, interpolate=True):
    """Crossing point of the isoline on one edge, by linear interpolation (or the edge midpoint)."""
    p, q = {0: (a, b), 1: (b, c), 2: (d, c), 3: (a, d)}[edge]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip((iso - p) / (q - p), 0.0, 1.0) if interpolate else np.full_like(p, 0.5)
    if edge == 0:
        return xj + t * (xj1 - xj), yi
    if edge == 2:
        return xj + t * (xj1 - xj), yi1
    if edge == 1:
        return xj1, yi + t * (yi1 - yi)
    return xj, yi + t * (yi1 - yi)


def marching_squares(F, x, y, iso, interpolate=True, return_stats=False):
    """Return isoline segments as an array of shape (n, 2, 2): [[x0, y0], [x1, y1]]."""
    F = np.asarray(F, dtype=float)
    a, b = F[:-1, :-1], F[:-1, 1:]
    d, c = F[1:, :-1], F[1:, 1:]
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
    case = (8 * (a >= iso) + 4 * (b >= iso) + 2 * (c >= iso) + 1 * (d >= iso)).astype(np.int8)
    case[~valid] = 0
    centre_above = (a + b + c + d) / 4.0 >= iso
    X, Y = np.meshgrid(x, y)
    xj, xj1 = X[:-1, :-1], X[:-1, 1:]
    yi, yi1 = Y[:-1, :-1], Y[1:, :-1]
    segs = []
    groups = [((k,), v) for k, v in CASE_SEGMENTS.items() if v]
    groups += [((k, ca), v) for (k, ca), v in SADDLE_SEGMENTS.items()]
    for key, seg_list in groups:
        if len(key) == 1:
            m = case == key[0]
        else:
            m = (case == key[0]) & (centre_above == key[1])
        if not m.any():
            continue
        args = (a[m], b[m], c[m], d[m], iso, xj[m], xj1[m], yi[m], yi1[m])
        for e0, e1 in seg_list:
            x0, y0 = _edge_points(e0, *args, interpolate=interpolate)
            x1, y1 = _edge_points(e1, *args, interpolate=interpolate)
            segs.append(np.stack([np.stack([x0, y0], -1), np.stack([x1, y1], -1)], 1))
    out = np.concatenate(segs) if segs else np.empty((0, 2, 2))
    if return_stats:
        counts = np.bincount(case[valid].ravel(), minlength=16)
        return out, counts
    return out


def segment_length_km(segs):
    """Total length of lon/lat segments in km (local equirectangular approximation, fine at 4 km spacing)."""
    if len(segs) == 0:
        return 0.0
    lat_m = np.deg2rad((segs[:, 0, 1] + segs[:, 1, 1]) / 2)
    dx = (segs[:, 1, 0] - segs[:, 0, 0]) * 111.32 * np.cos(lat_m)
    dy = (segs[:, 1, 1] - segs[:, 0, 1]) * 110.57
    return float(np.sum(np.hypot(dx, dy)))
