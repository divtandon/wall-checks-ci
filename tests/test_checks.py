"""Each test is a bug or trade-off I actually hit in my experiments, rebuilt on a synthetic 10 m x 6 m wall."""
import numpy as np

from wallcheck.checks import depth_along_normal, rebuild_triangles, sample_wall, window_by_depth, window_iou

WALL = (0.0, 0.0, 10.0, 6.0)
WINDOWS = [(1.0, 1.0, 2.5, 3.0), (6.0, 1.0, 7.5, 3.0)]
OUTWARD = (0.0, 1.0, 0.0)  # the wall lies in the x-z plane and faces +y


def synthetic_wall(window_depth=0.2, n=20000, seed=0):
    """Points on the wall; points inside a window sit window_depth metres behind the surface."""
    uv = sample_wall(WALL, n, seed)
    inside = np.zeros(n, bool)
    for u0, v0, u1, v1 in WINDOWS:
        inside |= (uv[:, 0] > u0) & (uv[:, 0] < u1) & (uv[:, 1] > v0) & (uv[:, 1] < v1)
    points = np.column_stack([uv[:, 0], np.where(inside, -window_depth, 0.0), uv[:, 1]])
    return points, inside


def test_iou_full_and_half_overlap():
    truth = np.array([1, 1, 1, 1, 0, 0])
    area = np.ones(6)
    assert window_iou(truth, truth, area) == 1.0
    assert window_iou([1, 1, 0, 0, 0, 0], truth, area) == 0.5
    assert window_iou(np.zeros(6), np.zeros(6), area) == 1.0


def test_iou_is_area_weighted():
    # one big wrong sample must count more than two small right ones
    assert window_iou([1, 1, 1], [1, 1, 0], np.array([1.0, 1.0, 8.0])) == 0.2


def test_rebuilt_wall_triangle_count():
    # a polygon with v vertices and h holes triangulates into v + 2h - 2 triangles: 12 + 4 - 2 = 14
    assert rebuild_triangles(WALL, WINDOWS) == (14, 4)


def test_recessed_windows_have_negative_depth():
    # my depth map once showed windows in front of the wall because the normal pointed inwards
    points, inside = synthetic_wall()
    depth = depth_along_normal(points, np.zeros(3), OUTWARD)
    assert np.allclose(depth[inside], -0.2)
    assert np.allclose(depth[~inside], 0.0)


def test_depth_rule_finds_the_windows():
    points, inside = synthetic_wall()
    labels = window_by_depth(depth_along_normal(points, np.zeros(3), OUTWARD))
    assert window_iou(labels, inside, np.ones(len(inside))) == 1.0


def test_deep_windows_fall_outside_the_30cm_band():
    # windows 35 cm deep were scored as wall by my 30 cm answer key; a 50 cm band keeps them
    points, inside = synthetic_wall(window_depth=0.35)
    depth = depth_along_normal(points, np.zeros(3), OUTWARD)
    area = np.ones(len(inside))
    assert window_iou(window_by_depth(depth, tol=0.30), inside, area) == 0.0
    assert window_iou(window_by_depth(depth, tol=0.50), inside, area) == 1.0


def test_same_seed_gives_same_points():
    # unseeded randomness once changed my scores by 0.2-0.4 between identical runs
    assert np.array_equal(sample_wall(WALL, 100, seed=7), sample_wall(WALL, 100, seed=7))
    assert not np.array_equal(sample_wall(WALL, 100, seed=7), sample_wall(WALL, 100, seed=8))
