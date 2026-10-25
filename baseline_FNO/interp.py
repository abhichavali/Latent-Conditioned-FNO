"""Motion-compensated frame interpolation, for smooth demo playback of coarse (one FNO step per frame) rollouts.

Linear crossfade between frames turns a crest that moves about its own width per step into two half-height ghost
crests. Instead: estimate a dense displacement field between consecutive frames (coarse-to-fine Horn-Schunck optical
flow on the primary field, periodic grid), warp both frames toward the intermediate time and blend.
"""

import jax
import jax.numpy as jnp
import numpy as np


def warp(I, fx, fy):
    """Backward warp out(x) = I(x - f), bilinear, periodic.  I (..., n, n); f in cells."""
    n = I.shape[-1]
    ys, xs = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    x, y = xs - fx, ys - fy
    x0, y0 = jnp.floor(x), jnp.floor(y)
    wx, wy = x - x0, y - y0
    x0, y0 = x0.astype(jnp.int32) % n, y0.astype(jnp.int32) % n
    x1, y1 = (x0 + 1) % n, (y0 + 1) % n
    g = lambda yy, xx: I[..., yy, xx]
    return (1 - wy) * ((1 - wx) * g(y0, x0) + wx * g(y0, x1)) + wy * ((1 - wx) * g(y1, x0) + wx * g(y1, x1))


def _blur(f, sigma):
    n = f.shape[-1]
    k = jnp.fft.fftfreq(n)
    g = jnp.exp(-2 * (jnp.pi * sigma) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))
    return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(f) * g))


def flow(I0, I1, R=10, win=3.0, lam=2e-3):
    """Displacement f (cells) with I1(x + f) ~ I0(x)."""
    s = jnp.std(I0) + jnp.std(I1) + 1e-12
    a, b = I0 / s, I1 / s
    d = jnp.arange(-R, R + 1)
    dy, dx = (g.ravel() for g in jnp.meshgrid(d, d, indexing="ij"))
    cost = jax.vmap(lambda y, x: _blur((jnp.roll(b, (-y, -x), (0, 1)) - a) ** 2, win) + lam * (x * x + y * y))(dy, dx)
    cost = cost.reshape(2 * R + 1, 2 * R + 1, *a.shape)
    i = jnp.argmin(cost.reshape(-1, *a.shape), 0)
    iy, ix = i // (2 * R + 1), i % (2 * R + 1)

    def sub(c_m, c_0, c_p):  # parabola vertex offset in [-0.5, 0.5]
        den = c_m - 2 * c_0 + c_p
        return jnp.clip(jnp.where(den > 1e-12, 0.5 * (c_m - c_p) / den, 0.0), -0.5, 0.5)

    at = lambda yy, xx: jnp.take_along_axis(cost.reshape(-1, *a.shape), (yy.clip(0, 2 * R) * (2 * R + 1) + xx.clip(0, 2 * R))[None], 0)[0]
    fx = ix - R + sub(at(iy, ix - 1), at(iy, ix), at(iy, ix + 1)) * ((ix > 0) & (ix < 2 * R))
    fy = iy - R + sub(at(iy - 1, ix), at(iy, ix), at(iy + 1, ix)) * ((iy > 0) & (iy < 2 * R))
    w = _blur((a - _blur(a, win)) ** 2 + (b - _blur(b, win)) ** 2, win) + 1e-3  # confidence: local structure
    return _blur(fx * w, win) / _blur(w, win), _blur(fy * w, win) / _blur(w, win)


@jax.jit
def _between(A, B, alphas):  # A, B (C, n, n); channel 0 drives the flow -> (k, C, n, n)
    fx, fy = flow(A[0], B[0])
    return jax.vmap(lambda t: (1 - t) * warp(A, t * fx, t * fy) + t * warp(B, -(1 - t) * fx, -(1 - t) * fy))(alphas)


def upsample(frames, k, method="flow"):
    """(T, C, n, n) -> ((T - 1) k + 1, C, n, n): k - 1 new frames between each pair."""
    frames = jnp.asarray(frames, jnp.float32)
    alphas = jnp.arange(k) / k
    if method == "linear":
        mid = (1 - alphas[None, :, None, None, None]) * frames[:-1, None] + alphas[None, :, None, None, None] * frames[1:, None]
    else:
        mid = jnp.stack([_between(frames[t], frames[t + 1], alphas) for t in range(len(frames) - 1)])
    return np.asarray(jnp.concatenate([mid.reshape(-1, *frames.shape[1:]), frames[-1:]]))


def check(fine, k):
    """Interpolate every k-th frame of a finely sampled truth back to full rate; rel L2 (channel 0) on new frames."""
    new = np.arange(len(fine)) % k != 0
    out = {}
    for m in ("linear", "flow"):
        up = upsample(fine[::k], k, m)[: len(fine)]
        d, y = (up - fine[: len(up)])[new[: len(up)], 0], fine[: len(up)][new[: len(up)], 0]
        out[m] = float(np.sqrt((d**2).sum() / (y**2).sum()))
    return out


if __name__ == "__main__":  # a bump moving 6 cells per frame: flow interpolation should beat crossfade by a lot
    n = 64
    ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    bump = lambda c: np.exp(-(((xs - c + n / 2) % n - n / 2) ** 2 + (ys - 32.0) ** 2) / (2 * 3.0**2))
    fine = np.stack([bump(10 + 1.5 * t)[None] for t in range(13)]).astype(np.float32)
    e = check(fine, 4)
    assert e["flow"] < 0.5 * e["linear"], e
    print("interp ok", e)
