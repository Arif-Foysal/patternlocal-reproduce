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
    """Apply Gaussian smoothing with a relative maximum-support threshold.

    The paper (App. C.1) uses a Gaussian filter with ``sigma_smooth = 1.5`` and
    a *maximum support threshold of 5%*: coefficients whose magnitude falls
    below 5% of the largest coefficient are discarded.  The threshold must be
    relative and sign-preserving -- an absolute ``smoothed < 0.05`` test would
    erase every negative lobe, which destroys the ``-aL`` component of XOR and
    RIGID signals.
    """
    smoothed = gaussian_filter(signal, sigma=sigma)
    peak = float(np.max(np.abs(smoothed)))
    if peak > 0:
        smoothed[np.abs(smoothed) < threshold * peak] = 0.0
    return smoothed


def apply_rigid_transform(signal, rng):
    """Apply random 90-degree rotation and translation (zero-padded, no wrap).

    The paper applies R(n) as a rigid-body transformation: rotation in 90-degree
    increments followed by a translation.  ``np.roll`` would wrap foreground
    pixels around the image border, which is not a rigid transform of the image
    content, so we shift with zero filling instead.
    """
    k = rng.integers(0, 4)
    rotated = np.rot90(signal, k)

    max_shift = signal.shape[0] // 4
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))

    translated = np.zeros_like(rotated)
    h, w = rotated.shape
    # Source and destination slices for an integer shift by (dy, dx).
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
    dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
    translated[dst_y0:dst_y1, dst_x0:dst_x1] = rotated[src_y0:src_y1, src_x0:src_x1]
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
                base = t_signal + l_signal
            else:
                base = t_signal - l_signal
            # Paper (App. C.1): each class also contains the global sign-flipped
            # variant, so the label is determined by the *relative* sign of the
            # two tetrominoes (a true XOR), not by one tetromino's absolute sign.
            #   y=0: aXOR++ = aT + aL   or   aXOR-- = -aT - aL
            #   y=1: aXOR+- = aT - aL   or   aXOR-+ = -aT + aL
            if rng.random() < 0.5:
                base = -base
            smoothed = smooth_signal(base)

        elif scenario == "RIGID":
            # Paper (Sec. 4.1 / App. C.1): *each* tetromino is randomly
            # translated and rotated by 90-degree increments, then combined with
            # the class-dependent sign; a global sign flip is applied as in XOR.
            t_signal = apply_rigid_transform(signal_cache[(yi, "T")].copy(), rng)
            l_signal = apply_rigid_transform(signal_cache[(yi, "L")].copy(), rng)
            if yi == 0:
                base = t_signal + l_signal
            else:
                base = t_signal - l_signal
            if rng.random() < 0.5:
                base = -base
            smoothed = smooth_signal(base)

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
