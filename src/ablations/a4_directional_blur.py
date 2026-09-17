"""A4 - flou directionnel (protocole §3, pipeline step 4).

A box filter along one axis only: theta=0 degrees smooths along the body
axis (birds are photographed lying horizontally, so that's the image's
x-axis), theta=90 degrees smooths across it (y-axis). Applied at image
level, before cropping, so the kernel always has real neighbourhood to
average over (a kernel up to 101px wide would replicate-pad almost half a
224px crop if applied after cropping - protocole §2 "Les quatre
contraintes").

Drawn once per sac (protocole "tire par sac"): both angle and length are
shared by all 16 crops.

This removes structure in one direction, but also removes real variance/
energy in the process - it is not a pure "direction" ablation, see
protocole §3 "A4 - elle retire de la direction, mais aussi de l'energie".
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

CODE = "A4"
THETA_CHOICES = (0, 90)
LENGTH_MIN = 17
LENGTH_MAX = 101


def draw_params(rng: np.random.Generator) -> Dict[str, int]:
    theta = int(THETA_CHOICES[rng.integers(2)])
    # log-uniform over an odd length in [17, 101] - draw log-uniform in the
    # continuous range then round to the nearest odd integer.
    log_l = rng.uniform(np.log(LENGTH_MIN), np.log(LENGTH_MAX))
    length = int(round(np.exp(log_l)))
    if length % 2 == 0:
        length += 1
    length = int(np.clip(length, LENGTH_MIN, LENGTH_MAX))
    return {"theta": theta, "length": length}


def apply(rgb_uint8: np.ndarray, theta: int, length: int) -> np.ndarray:
    if theta == 0:
        kernel = np.full((1, length), 1.0 / length, dtype=np.float32)
    else:
        kernel = np.full((length, 1), 1.0 / length, dtype=np.float32)
    blurred = cv2.filter2D(rgb_uint8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    return blurred


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    params = draw_params(rng)
    blurred = apply(canvas, **params)

    print(f"A4: params drawn = {params}")
    print(f"pixel std before: {canvas.std():.1f}  after: {blurred.std():.1f} (blur should reduce it)")
