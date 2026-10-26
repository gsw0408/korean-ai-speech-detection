"""64,000-sample inference windows; short inputs are rejected without padding."""
import numpy as np

WINDOW = 64000
HOP = 32000


def starts(n):
    if n < WINDOW:
        raise ValueError('shorter than the v2 detector window')
    offsets = list(range(0, n - WINDOW + 1, HOP))
    if offsets[-1] != n - WINDOW:
        offsets.append(n - WINDOW)
    return offsets


def windows(samples):
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('invalid front-end output')
    return [x[s:s + WINDOW] for s in starts(len(x))]
