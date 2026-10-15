"""XAI-TRIS benchmark data generator following Appendix C.1 of the paper.

Each instance: x = alpha * (R o (H o a)) + (1-alpha) * (G o eta)
where:
  - a is the class-dependent tetromino signal
  - H is Gaussian smoothing (sigma=1.5, threshold 5%)
  - G is identity (WHITE) or Gaussian smoothing (sigma=10) for CORR
  - eta ~ N(0, I)
  - alpha controls signal-to-noise ratio

Ground truth F+ (Eq. 13/14) defines important pixels.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def make_tetromino(name, size=64):
    """Create a tetromino shape on a size x size grid.

    T-shape (top-left for y=0):
        XXX
         X
         X

    L-shape (bottom-right for y=1):
        X
        X
        XX
    """
    grid = np.zeros((size, size), dtype=np.float64)
    cell = size // 8

    if name == "T":
        top = size // 8
        left = size // 8
        grid[top:top+cell, left:left+3*cell] = 1.0
        grid[top+cell:top+3*cell, left+cell:left+2*cell] = 1.0
    elif name == "L":
        bottom = size - size // 8
        right = size - size // 8
        grid[bottom-3*cell:bottom, right-cell:right] = 1.0
        grid[bottom-cell:bottom, right-3*cell:right-cell] = 1.0
    else:
        raise ValueError(f"Unknown tetromino: {name}")

    return grid


def smooth_signal(signal, sigma=1.5, threshold=0.05):
    """Apply Gaussian smoothing with threshold."""
    smoothed = gaussian_filter(signal, sigma=sigma)
    smoothed[smoothed < threshold] = 0.0
    return smoothed


def apply_rigid_transform(signal, rng):
    """Apply random 90-degree rotation and translation."""
    k = rng.integers(0, 4)
    rotated = np.rot90(signal, k)

    max_shift = signal.shape[0] // 4
    dx = rng.integers(-max_shift, max_shift + 1)
    dy = rng.integers(-max_shift, max_shift + 1)
    translated = np.roll(rotated, shift=(dy, dx), axis=(0, 1))
    return translated


def generate_xai_tris(
    n_samples=40000,
    scenario="LIN",
    noise="WHITE",
    alpha=0.5,
    image_size=64,
    seed=42,
):
    """Generate XAI-TRIS benchmark dataset.

    Args:
        n_samples: Number of samples to generate
        scenario: 'LIN', 'XOR', or 'RIGID'
        noise: 'WHITE' or 'CORR'
        alpha: Signal-to-noise mixing coefficient [0, 1]
        image_size: Size of square images (default 64)
        seed: Random seed for reproducibility

    Returns:
        X: Flattened images, shape (n_samples, image_size*image_size)
        y: Binary labels, shape (n_samples,)
        masks: Ground truth attribution masks, shape (n_samples, image_size*image_size)
    """
    rng = np.random.default_rng(seed)
    d = image_size * image_size

    X = np.zeros((n_samples, d), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.int64)
    masks = np.zeros((n_samples, d), dtype=np.float64)

    signal_cache = {}
    for s in [0, 1]:
        signal_cache[(s, "T")] = make_tetromino("T", image_size)
        signal_cache[(s, "L")] = make_tetromino("L", image_size)

    for i in range(n_samples):
        yi = rng.integers(0, 2)
        y[i] = yi

        if scenario == "LIN":
            if yi == 0:
                signal = signal_cache[(0, "T")].copy()
            else:
                signal = signal_cache[(1, "L")].copy()
            smoothed = smooth_signal(signal)

        elif scenario == "XOR":
            t_signal = signal_cache[(yi, "T")].copy()
            l_signal = signal_cache[(yi, "L")].copy()
            if yi == 0:
                signal = t_signal + l_signal
            else:
                signal = t_signal - l_signal
            smoothed = smooth_signal(signal)

        elif scenario == "RIGID":
            t_signal = signal_cache[(yi, "T")].copy()
            l_signal = signal_cache[(yi, "L")].copy()
            if yi == 0:
                signal = t_signal + l_signal
            else:
                signal = t_signal - l_signal
            rigid = apply_rigid_transform(signal, rng)
            smoothed = smooth_signal(rigid)

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        noise_sample = rng.standard_normal((image_size, image_size))
        if noise == "CORR":
            noise_sample = gaussian_filter(noise_sample, sigma=10.0)
        elif noise != "WHITE":
            raise ValueError(f"Unknown noise type: {noise}")

        signal_norm = np.linalg.norm(smoothed.ravel())
        noise_norm = np.linalg.norm(noise_sample.ravel())
        if signal_norm > 0:
            smoothed_normed = smoothed * (1.0 / signal_norm)
        else:
            smoothed_normed = smoothed
        noise_normed = noise_sample * (1.0 / noise_norm) if noise_norm > 0 else noise_sample

        image = alpha * smoothed_normed + (1 - alpha) * noise_normed
        X[i] = image.ravel()

        mask = (np.abs(alpha * smoothed_normed) > 0).astype(np.float64)
        if scenario == "LIN":
            t_mask = smooth_signal(signal_cache[(yi, "T")]) > 0
            l_mask = smooth_signal(signal_cache[(yi, "L")]) > 0
            mask = (t_mask | l_mask).astype(np.float64)
        masks[i] = mask.ravel()

    return X, y, masks


def split_dataset(X, y, masks, train_ratio=0.9, val_ratio=0.05, seed=42):
    """Split dataset into train/val/test sets."""
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        (X[train_idx], y[train_idx], masks[train_idx]),
        (X[val_idx], y[val_idx], masks[val_idx]),
        (X[test_idx], y[test_idx], masks[test_idx]),
    )
