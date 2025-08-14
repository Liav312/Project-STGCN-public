import numpy as np
from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RandomFlipAngles(BaseTransform):
    

    def __init__(self, prob=0.5):
        self.prob = prob
        self.swap_pairs = [
            (0, 8), (1, 9), (2, 10), (3, 11),
            (4, 12), (5, 13), (6, 14), (7, 15)
        ]

    def transform(self, results):
        if np.random.rand() < self.prob:
            kp = results['keypoint']
            kp = kp.copy()
            for i, j in self.swap_pairs:
                kp[..., (i, j), :] = kp[..., (j, i), :]
            results['keypoint'] = kp
        return results


@TRANSFORMS.register_module()
class AddNoise(BaseTransform):
    

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def transform(self, results):
        kp = results['keypoint']
        noise = np.random.normal(0, self.sigma, size=kp.shape).astype(kp.dtype)
        results['keypoint'] = kp + noise
        return results
    
@TRANSFORMS.register_module()
class AngleJitter(BaseTransform):
    """Randomly jitter angles by a small offset."""

    def __init__(self, offset: float = 5.0):
        self.offset = offset / 180.0  # Normalize to [0,1] range

    def transform(self, results):
        kp = results['keypoint']
        jitter = np.random.uniform(-self.offset, self.offset, size=kp.shape)
        results['keypoint'] = kp + jitter.astype(kp.dtype)
        return results

@TRANSFORMS.register_module()
class ContrastiveAug(BaseTransform):
    def __init__(self,
                 rom_range=(0.1, 1.2),
                 stretch_range=(0.5, 1.5),
                 sigma_deg=5.0,
                 p_mirror=0.5,
                 p_identity=0.05):
        self.rom_range = rom_range
        self.stretch_range = stretch_range
        self.sigma_deg = sigma_deg
        self.p_mirror = p_mirror
        self.p_identity = p_identity
        self.swap_pairs = [
            (0, 8), (1, 9), (2, 10), (3, 11),
            (4, 12), (5, 13), (6, 14), (7, 15)
        ]

    def _time_stretch(self, arr: np.ndarray, beta: float) -> np.ndarray:
        T, V = arr.shape
        new_len = max(1, int(round(T * beta)))
        idx = np.linspace(0, T - 1, new_len)
        stretched = np.zeros((new_len, V), dtype=arr.dtype)
        for v in range(V):
            stretched[:, v] = np.interp(idx, np.arange(T), arr[:, v])
        if new_len < T:
            pre = (T - new_len) // 2
            post = T - new_len - pre
            stretched = np.pad(stretched, ((pre, post), (0, 0)), mode='edge')
        elif new_len > T:
            stretched = stretched[:T]
        return stretched

    def transform(self, results):
        if np.random.rand() < self.p_identity:
            return results

        # ``keypoint`` has shape (M,T,V,C) = (1, window, 16, 2) with cos/sin
        kp = results['keypoint'][0]  # (T, V, 2)
        cos = kp[..., 0]
        sin = kp[..., 1]

        # Recover angles using atan2, assuming sin and cos represent components of vectors derived from XYZ
        theta = np.arctan2(sin, cos)  # (T, V)

        # Apply ROM scaling to angles to simulate variations in range of motion
        alpha = np.random.uniform(*self.rom_range)
        theta *= alpha

        # Apply time stretch to the angle sequences
        beta = np.random.uniform(*self.stretch_range)
        theta = self._time_stretch(theta, beta)

        # Apply angle jitter in radians to simulate small perturbations, robust for real XYZ-derived vectors
        sigma_rad = self.sigma_deg * np.pi / 180.0
        theta += np.random.normal(0, sigma_rad, size=theta.shape).astype(theta.dtype)

        # Apply mirroring: swap joints and negate angles to simulate left-right flip, appropriate for symmetric joints in XYZ space
        if np.random.rand() < self.p_mirror:
            for i, j in self.swap_pairs:
                theta[:, i], theta[:, j] = -theta[:, j], -theta[:, i]

        # Recompute sin and cos from augmented angles to maintain unit norm
        cos_aug = np.cos(theta)
        sin_aug = np.sin(theta)

        aug = np.stack([cos_aug, sin_aug], axis=-1)
        results['keypoint'] = aug[None, :, :, :]
        return results
