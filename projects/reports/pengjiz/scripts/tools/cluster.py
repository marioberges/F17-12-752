import numpy as np
from typing import Tuple, Iterable, Dict, Any
from functools import reduce, partial
from itertools import combinations
from .helpers import segmented_map


def _assign(vec, break_pts: Iterable):
    return reduce(lambda x, y: x + (vec > y).astype(int),
                  break_pts,
                  np.zeros_like(vec))


def _variance(vec, groups: Iterable) -> float:
    return sum(segmented_map(vec, groups,
                             lambda x: ((x - x.mean()) ** 2).sum())
               .values())


# Unfinished, do not use.
def jenks(vec, num_cls: int) -> Tuple[Any, Dict[int, float]]:
    sorted_vec = np.sort(np.unique(vec))
    break_ptss = combinations(sorted_vec, num_cls - 1)
    groupss = map(partial(_assign, vec), break_ptss)
    best_group = min(groupss, key=partial(_variance, vec))
    centroids = segmented_map(vec, best_group, lambda x: x.mean())
    return best_group, centroids


def cassign1d(vec, centroids):
    def fn_dists(centroid: float):
        return (vec - centroid) ** 2

    dist_mat = np.array([*map(fn_dists, centroids)])
    return dist_mat.argmin(axis=0)


def relabel(labels: Iterable, centroids: Iterable) -> Iterable:
    order = np.argsort(centroids)
    return [order[g] for g in labels]
