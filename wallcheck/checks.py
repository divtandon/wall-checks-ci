"""Small checks that mirror real bugs from my facade experiments, run on synthetic walls only (no project data)."""
import numpy as np
import shapely


def window_iou(pred, truth, area):
    """Area-weighted IoU of the window label between two per-sample labelings (1.0 when both are empty)."""
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    union = area[pred | truth].sum()
    return float(area[pred & truth].sum() / union) if union > 0 else 1.0


def rebuild_triangles(wall, windows):
    """Rebuild a flat wall with rectangular window holes: (triangles in the wall, triangles in the windows)."""
    holes = [shapely.box(*b) for b in windows]
    face = shapely.Polygon(shapely.box(*wall).exterior.coords, [h.exterior.coords for h in holes])
    count = lambda g: len(shapely.get_parts(shapely.constrained_delaunay_triangles(g)))
    return count(face), sum(count(h) for h in holes)


def depth_along_normal(points, wall_point, outward_normal):
    """Signed depth of each point: negative = behind the wall surface (recessed), positive = in front."""
    n = np.asarray(outward_normal, float)
    n = n / np.linalg.norm(n)
    return (np.asarray(points, float) - wall_point) @ n


def window_by_depth(depth, recess_min=0.05, tol=0.30):
    """Depth rule: a sample is a window if it sits between recess_min and tol metres behind the wall."""
    return (depth <= -recess_min) & (depth >= -tol)


def sample_wall(wall, n, seed):
    """Random points on a wall rectangle (u0, v0, u1, v1); the same seed must give the same points."""
    rng = np.random.default_rng(seed)
    u0, v0, u1, v1 = wall
    return np.column_stack([rng.uniform(u0, u1, n), rng.uniform(v0, v1, n)])
