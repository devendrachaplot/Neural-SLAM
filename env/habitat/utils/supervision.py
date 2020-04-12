import numpy as np


class HabitatMaps(object):
    def __init__(self, env, N=int(1e6), resolution=5, padding=0):
        # convert to cm
        self.resolution = resolution
        self.padding = padding

        pts = self._sample_points(env, N) * 100.

        # Bin points based on x and z values, so that
        # we can quickly pool them based on y-filtering.
        self.y = pts[:, 1]
        zx = pts[:, [2, 0]]
        self.origin, self.size, self.max = self._make_map(
            zx, self.padding, self.resolution)

        zx = zx - self.origin
        self.zx = (zx / self.resolution).astype(np.int)

    def get_map(self, y, lb, ub):
        ids = np.logical_and(self.y > y + lb, self.y < y + ub)
        num_points = np.zeros((self.size[1], self.size[0]), dtype=np.int32)
        np.add.at(num_points, (self.zx[ids, 1], self.zx[ids, 0]), 1)
        return num_points

    def _make_map(self, zx, padding, resolution):
        """Returns a map structure."""
        min_, max_ = self._get_xy_bounding_box(zx, padding=padding)
        sz = np.ceil((max_ - min_ + 1) / resolution).astype(np.int32)
        max_ = min_ + sz * resolution - 1
        return min_, sz, max_

    def _get_xy_bounding_box(self, zx, padding):
        """Returns the xy bounding box of the environment."""
        min_ = np.floor(np.min(zx, axis=0) - padding).astype(np.int)
        max_ = np.ceil(np.max(zx, axis=0) + padding).astype(np.int)
        return min_, max_

    def _sample_points(self, env, N):
        pts = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            pts[i, :] = env.sim.sample_navigable_point()
        return pts
