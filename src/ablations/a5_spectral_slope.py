"""A5 - pente du spectre de Fourier (protocole §3 et §4 "constantes remesurees").

Regrades the radial amplitude spectrum of the crop towards a fixed target
slope alpha0, phase unchanged, same family of tools as SHINE's sfMatch
(Willenbockel et al. 2010, cited in the protocol's references). Concretely:
every frequency bin's magnitude is multiplied by a positive real gain, so
phase is untouched by construction, and the gain is the same for R, G and B
- it is derived once from the G channel's own radial profile (protocole
§3/§8: "ajustee sur le canal G").

Two target slopes, not one, because the 128px crops are resampled to 224px
before reaching this step and that resampling itself changes the apparent
spectral slope (protocole §4 explains why at length: the median blur k=15
becomes an effective 26px kernel after the 128->224 resize, which is why the
128-crop target is steeper). Both numbers are given directly by the
protocol (measured on 120 specimens, band 2-9 cycles/crop) - not re-derived
here.

The target actually used is that reference value plus a small uniform
jitter (+/- ALPHA_JITTER), drawn fresh per crop: apply()'s regrade hits its
own target almost exactly in float precision, but the final uint8
clip/quantize step can leave a small, reproducible residual
(alpha_after - target_alpha) on images with more clipping (measured on 150
real crops: std ~0.005, worst case ~0.025 out of a target of ~1.49-2.19) -
a fixed target would make that residual a fixed function of image content,
which is exactly the kind of stable, potentially class-correlated leftover
signal an ablation is supposed to remove, not introduce. Jittering the
target decorrelates the residual from image content the same way A3's
severity and A4's angle/length are drawn randomly rather than fixed,
instead of leaving A5 as the one CIELAB/spectral-group ablation with no
drawn parameter at all.

Applied per crop (never per sac), after A3 if both are drawn (protocole §8:
A3 does not touch the histogram, so A5 sees whatever spectrum A3 left it,
which is the intended behaviour).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

CODE = "A5"
BAND_LOW = 2.0
BAND_HIGH = 9.0
TARGET_ALPHA_224 = 1.49
TARGET_ALPHA_128 = 2.19
ALPHA_JITTER = 0.1  # +/- this, uniform, added to the reference target - see module docstring
ANCHOR_RADIUS = BAND_LOW  # the gain is anchored so it leaves this radius untouched


def _radial_index(size: int) -> np.ndarray:
    freqs = np.fft.fftshift(np.fft.fftfreq(size) * size)
    fy, fx = np.meshgrid(freqs, freqs, indexing="ij")
    radius = np.sqrt(fx**2 + fy**2)
    return np.round(radius).astype(int)


def _radial_average(magnitude: np.ndarray, radial_idx: np.ndarray, max_r: int) -> np.ndarray:
    sums = np.bincount(radial_idx.ravel(), weights=magnitude.ravel(), minlength=max_r + 1)
    counts = np.bincount(radial_idx.ravel(), minlength=max_r + 1)
    counts[counts == 0] = 1
    return sums / counts


def measure_alpha(crop_uint8: np.ndarray, band=(BAND_LOW, BAND_HIGH)) -> float:
    """Fit the current radial-average slope of the G channel in the given
    band by ordinary least squares in log-log space. Used for logging/
    diagnostics (figures, benchmark script), not by apply() itself."""
    g = crop_uint8[..., 1].astype(np.float64)
    size = g.shape[0]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    radial_idx = _radial_index(size)
    max_r = int(radial_idx.max())
    profile = _radial_average(spectrum, radial_idx, max_r)

    radii = np.arange(max_r + 1)
    in_band = (radii >= band[0]) & (radii <= band[1]) & (profile > 0)
    log_r = np.log(radii[in_band])
    log_a = np.log(profile[in_band])
    slope, _intercept = np.polyfit(log_r, log_a, 1)
    return float(-slope)


def apply(crop_uint8: np.ndarray, target_alpha: float) -> Tuple[np.ndarray, Dict[str, float]]:
    size = crop_uint8.shape[0]
    radial_idx = _radial_index(size)
    max_r = int(radial_idx.max())

    g = crop_uint8[..., 1].astype(np.float64)
    g_fft = np.fft.fftshift(np.fft.fft2(g))
    magnitude = np.abs(g_fft)
    profile = _radial_average(magnitude, radial_idx, max_r)

    anchor_r = int(round(ANCHOR_RADIUS))
    anchor_amplitude = profile[anchor_r] if profile[anchor_r] > 0 else 1.0
    radii = np.arange(max_r + 1).astype(np.float64)
    with np.errstate(divide="ignore"):
        target_profile = anchor_amplitude * np.power(radii / anchor_r, -target_alpha)
    target_profile[0] = profile[0]  # never touch the DC term (mean brightness)

    gain_1d = np.ones_like(profile)
    nonzero = profile > 0
    gain_1d[nonzero] = target_profile[nonzero] / profile[nonzero]
    gain_2d = gain_1d[radial_idx]

    out = np.empty_like(crop_uint8, dtype=np.float64)
    for channel in range(3):
        channel_fft = np.fft.fftshift(np.fft.fft2(crop_uint8[..., channel].astype(np.float64)))
        regraded = channel_fft * gain_2d  # real, positive gain -> phase unchanged
        out[..., channel] = np.real(np.fft.ifft2(np.fft.ifftshift(regraded)))

    out_uint8 = np.clip(out, 0, 255).astype(np.uint8)
    meta = {
        "target_alpha": target_alpha,
        "alpha_before": measure_alpha(crop_uint8),
        "alpha_after": measure_alpha(out_uint8),
    }
    return out_uint8, meta


def target_alpha_for_crop_source(source_size: int) -> float:
    """The protocol's own reference value (measured on 120 specimens) for
    this crop size - the *centre* draw_target_alpha() jitters around, not
    what apply() is actually called with anymore (see module docstring)."""
    if source_size == 224:
        return TARGET_ALPHA_224
    if source_size == 128:
        return TARGET_ALPHA_128
    raise ValueError(f"A5 only has a target for 128/224 source crops, got {source_size}")


def draw_target_alpha(source_size: int, rng: np.random.Generator) -> float:
    """The reference target for this crop size, jittered by
    +/- ALPHA_JITTER (uniform, drawn fresh per crop) - what chain.py
    actually calls apply() with. See module docstring for why."""
    base = target_alpha_for_crop_source(source_size)
    return base + float(rng.uniform(-ALPHA_JITTER, ALPHA_JITTER))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # blocky upsampled noise, not pure white noise - it has real low-frequency
    # structure, so it has a measurable slope to regrade in the first place
    # (pure white noise is already flat: alpha ~ 0).
    base = rng.integers(0, 256, size=(28, 28, 3)).astype(np.uint8)
    crop = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))

    target = draw_target_alpha(224, rng)
    out, meta = apply(crop, target)

    print(f"A5: reference target (224px crop) = {target_alpha_for_crop_source(224)}  drawn target = {target:.3f}")
    print(f"alpha before: {meta['alpha_before']:.2f}  ->  after: {meta['alpha_after']:.2f}")
