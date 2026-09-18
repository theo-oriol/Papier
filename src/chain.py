"""The canonical transformation chain (protocole §2): one fixed pipeline,
ablations tapped in at four insertion points. This is the only place that
knows the *order* of operations - every ablation file only knows how to do
its own thing.

Pipeline, exactly as ordered in the protocol:
    0. read source (or A1/A2/A7's offline-precomputed source, if one of
       them is active - all three are deterministic given the specimen,
       see each module's docstring for why they're cached instead of
       computed online)
    1. prepare: build the mask, apply A6 if it's the active CIELAB-exclusive
       ablation (A1/A2/A7 already happened at step 0 - only A6 needs an
       online step 1, because its category is drawn per sac)
    2. vertical flip, p=0.5 (the one augmentation that is *not* an ablation)
    3. median blur, k=15
    4. A4 (directional blur), if active - image level
    5. sample crop positions (8x128 + 8x224)
    6. crop + resize (128 -> 224 bicubic; 224 stays native)
    7. A3 (block permutation), if active - per crop
    8. A5 (spectral slope), if active - per crop
    9. DINOv3 normalisation - done in the Dataset, not here (this module
       hands back uint8 crops; ToTensor+Normalize is a torchvision concern)

A single sac shares one flip draw, one set of 16 positions, and (if drawn)
one A4 kernel and one A6 category. A3's arrangement and A5 are per crop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from . import npy_io
from .ablations import a1_gray_lstar, a2_gray_standardized, a3_block_permutation
from .ablations import a4_directional_blur, a5_spectral_slope, a6_colour_removal, a7_chroma_only
from .ablations.common import Condition
from .sampling import load_candidates, sample_positions

MEDIAN_BLUR_K = 15
N_CROPS_PER_SIZE = 8
CROP_SIZES = (128, 224)


def _blur_near_crops(
    canvas: np.ndarray, crop_specs, k: int, a4_params: Dict[str, int] = None
) -> Tuple[np.ndarray, float, float]:
    """Median-blur (and, if `a4_params` is given, apply A4's directional
    blur too) only within each crop's own padded neighbourhood instead of
    touching the whole canvas. Exact, not an approximation: both operations
    are purely local (median blur: every output pixel depends only on its
    own k x k neighbourhood; A4: a 1D box filter along one axis only), so
    extracting a padded sub-block, running both filters on it in the same
    order the full-canvas pipeline would, and keeping only its interior
    reproduces exactly what the full-canvas computation would have given at
    those same pixels - verified bit-identical against the full-canvas
    version across both A4 orientations, several lengths, and crops at/near
    the canvas edge. The interior never sees the sub-block's own artificial
    edge (or, where the sub-block's edge coincides with the canvas's own
    edge, cv2 replicates/reflects identically there regardless of which
    array it was called on).

    Padding: median blur alone needs k//2 in every direction. A4 (if given)
    additionally needs length//2 *in its own blur axis only* (x for
    theta=0, y for theta=90) - plus another k//2 beyond that, since the
    pixel A4 reads at the edge of its own window must itself already be
    correctly median-blurred, which needs its own k x k neighbourhood of
    real data. Only correct for callers where nothing downstream ever reads
    a canvas pixel outside the given crops (see build_bag()).

    Returns (canvas_with_crops_blurred, seconds_spent_on_median, seconds_spent_on_a4).
    """
    median_pad = k // 2
    if a4_params is not None:
        theta, length = a4_params["theta"], a4_params["length"]
        extra = length // 2
        y_pad = median_pad + (extra if theta == 90 else 0)
        x_pad = median_pad + (extra if theta == 0 else 0)
    else:
        y_pad = x_pad = median_pad

    out = canvas.copy()
    h, w = canvas.shape[:2]
    t_median = t_a4 = 0.0
    for size, y, x in crop_specs:
        y0, y1 = max(0, y - y_pad), min(h, y + size + y_pad)
        x0, x1 = max(0, x - x_pad), min(w, x + size + x_pad)

        t0 = time.perf_counter()
        block = cv2.medianBlur(canvas[y0:y1, x0:x1], k)
        t_median += time.perf_counter() - t0

        if a4_params is not None:
            t0 = time.perf_counter()
            block = a4_directional_blur.apply(block, theta=theta, length=length)
            t_a4 += time.perf_counter() - t0

        iy, ix = y - y0, x - x0
        out[y : y + size, x : x + size] = block[iy : iy + size, ix : ix + size]
    return out, t_median, t_a4


def build_bag(
    dataset_dir: Path,
    dataset_a1_dir: Path,
    dataset_a2_dir: Path,
    dataset_a7_dir: Path,
    stem: str,
    condition: Condition,
    rng: np.random.Generator,
    *,
    fixed_a3_side: int = None,
    fixed_a4_params: Dict[str, int] = None,
    fixed_a6_category: str = None,
    positions_rng: np.random.Generator = None,
    dataset_a6_class_dir: Path = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build one sac's 16 crops (8x128 resized to 224, then 8x224 native,
    same order as the dataset's own candidate tables) under the given
    condition. Returns a (16, 224, 224, 3) uint8 array plus metadata about
    every draw that was made (useful for logging and for the eval script's
    per-condition bookkeeping).

    `positions_rng` lets the caller force the flip + crop positions to be
    drawn from an rng that depends only on the image, not the condition -
    required at evaluation time (protocole §7) so all conditions see the
    same geometry. Training just reuses `rng` for everything.

    The three `fixed_*` arguments pin a parameter that would otherwise be
    drawn from `rng`, for the evaluation grid (protocole §7's "ce qui est
    fige dans une condition d'evaluation"): a given h/angle-length/category
    stops being random and becomes part of the condition itself.

    `dataset_a6_class_dir` is optional and defaults to None (always use the
    online classifier, today's only behaviour). When given, and *both* the
    offline classification cache (scripts/build_a6_classification_cache.py)
    and A1's own cache have an entry for this specimen, A6 is resolved from
    those two caches instead of reclassifying online - see
    a6_colour_removal.apply_from_cache()'s docstring for why that's exact,
    not an approximation. Either cache being incomplete for this specimen
    (a FileNotFoundError from either read) falls back to the online
    classifier silently - callers don't need the caches to be fully built
    before turning this on.
    """
    geometry_rng = positions_rng if positions_rng is not None else rng
    meta: Dict[str, object] = {"stem": stem, "condition": str(condition)}
    # wall-clock cost of each stage, in seconds - a stage that wasn't drawn
    # for this sac (e.g. A4 when "A4" not in condition) simply has no key,
    # rather than a misleading 0.0 for "ran and took no time".
    timings: Dict[str, float] = {}
    meta["timings"] = timings
    t_bag_start = time.perf_counter()

    # --- steps 0-1 ---
    canvas, mask = read_working_image(
        dataset_dir, dataset_a1_dir, dataset_a2_dir, dataset_a7_dir, stem, condition, rng, meta,
        fixed_a6_category=fixed_a6_category,
        dataset_a6_class_dir=dataset_a6_class_dir,
    )

    # --- step 2: vertical flip, shared by the whole sac - drawn here but
    # *applied* below, after step 3's median blur, since a symmetric
    # kernel's median blur commutes exactly with a vertical flip (border
    # replication included - verified bit-identical on real-shaped data):
    # blurring the still-unflipped, C-contiguous canvas and flipping the
    # result afterward gives the same output while avoiding cv2.medianBlur's
    # ~10% slowdown on a flipped (negative-stride) view. ---
    flip = bool(geometry_rng.random() < 0.5)
    meta["flip"] = flip

    # --- step 5, drawn early (before step 3's blur, still logically step 5
    # in the canonical order - only *when* the RNG is consumed moved, not
    # what the pipeline does): crop positions only depend on the (blur- and
    # A4-invariant) candidate table, never on pixel values, so sampling them
    # now - instead of after blur/A4 - costs nothing and tells step 3 below
    # exactly which regions of the canvas the sac will ever actually read
    # from. Positions are in the canvas's own (pre-flip) coordinates here;
    # remapped for flip right before the crop loop, same as before. ---
    t0 = time.perf_counter()
    crop_specs_unflipped = []  # (size, y, x), pre-flip coordinates
    for size in CROP_SIZES:
        candidates = load_candidates(dataset_dir, stem, size)
        positions = sample_positions(candidates, N_CROPS_PER_SIZE, geometry_rng)
        crop_specs_unflipped.extend((size, y, x) for y, x in positions)
    timings["sample_positions_s"] = time.perf_counter() - t0

    # --- step 4's params, drawn early for the same reason as positions
    # above: A4 (if drawn) needs to be fused with step 3's blur below (both
    # are purely local ops - see _blur_near_crops()'s docstring - and A4
    # itself also commutes exactly with the later flip, verified
    # bit-identical, same as median blur), so its params must be known
    # before that fused call runs. ---
    a4_params = None
    if "A4" in condition:
        a4_params = fixed_a4_params if fixed_a4_params is not None else a4_directional_blur.draw_params(rng)
        meta["A4"] = a4_params

    # --- steps 3-4 fused: median blur, then A4 if drawn, both restricted to
    # each crop's own padded neighbourhood - nothing downstream ever reads a
    # canvas pixel outside the 16 crops' own footprints, so this is exact,
    # not an approximation, and skips processing the ~85%+ of the canvas no
    # crop ever touches (median blur unconditionally; A4 too, when drawn -
    # previously A4 forced a full-canvas pass for both steps). ---
    canvas, t_median, t_a4 = _blur_near_crops(canvas, crop_specs_unflipped, MEDIAN_BLUR_K, a4_params)
    timings["median_blur_s"] = t_median
    if a4_params is not None:
        timings["A4_s"] = t_a4

    t0 = time.perf_counter()
    if flip:
        canvas = np.flipud(canvas)
        mask = np.flipud(mask)
    timings["flip_s"] = time.perf_counter() - t0

    # remap for flip (y -> H-P-y is exact for a vertical permutation, so the
    # un-flipped positions above are transformed instead of re-derived)
    canvas_h = canvas.shape[0]
    if flip:
        crop_specs = [(size, canvas_h - size - y, x) for size, y, x in crop_specs_unflipped]
    else:
        crop_specs = crop_specs_unflipped
    meta["positions"] = crop_specs

    # --- steps 6-8: crop, resize, A3, A5 ---
    crops = []
    crop_meta = []
    t_crop_extract = t_a3 = t_a5 = 0.0
    for size, y, x in crop_specs:
        t0 = time.perf_counter()
        crop = canvas[y : y + size, x : x + size, :3].copy()
        if size == 128:
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        t_crop_extract += time.perf_counter() - t0

        this_crop_meta = {"source_size": size, "y": y, "x": x}

        if "A3" in condition:
            t0 = time.perf_counter()
            h = fixed_a3_side if fixed_a3_side is not None else a3_block_permutation.draw_side(rng)
            crop = a3_block_permutation.apply(crop, h, rng)
            this_crop_meta["A3_h"] = h
            t_a3 += time.perf_counter() - t0

        if "A5" in condition:
            t0 = time.perf_counter()
            target_alpha = a5_spectral_slope.draw_target_alpha(size, rng)
            crop, a5_meta = a5_spectral_slope.apply(crop, target_alpha)
            this_crop_meta["A5"] = a5_meta
            t_a5 += time.perf_counter() - t0

        crops.append(crop)
        crop_meta.append(this_crop_meta)

    meta["crops"] = crop_meta
    timings["crop_extract_resize_s"] = t_crop_extract  # all 16 crops combined, baseline cost (no ablation)
    if "A3" in condition:
        timings["A3_s"] = t_a3  # all 16 crops combined
    if "A5" in condition:
        timings["A5_s"] = t_a5  # all 16 crops combined
    timings["total_s"] = time.perf_counter() - t_bag_start
    return np.stack(crops, axis=0), meta


def read_working_image(
    dataset_dir: Path,
    dataset_a1_dir: Path,
    dataset_a2_dir: Path,
    dataset_a7_dir: Path,
    stem: str,
    condition: Condition,
    rng: np.random.Generator,
    meta: Dict[str, object],
    fixed_a6_category: str = None,
    dataset_a6_class_dir: Path = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Steps 0-1: get the (canvas, mask) this sac will be cropped from.

    A1/A2/A7 are read straight from their offline cache (they replace the
    source read itself, step 0 - each is a deterministic function of the
    specimen, so there's nothing to compute online). A6 still draws its
    category per sac, so it can't be precomputed the same way - but *given*
    a category, its removal is exact given the offline classification cache
    plus A1's own cache (see a6_colour_removal.apply_from_cache()); when
    `dataset_a6_class_dir` is given and both caches have this specimen,
    that replaces the online classifier below.
    """
    timings = meta["timings"]
    member = condition.cielab_member
    if member == "A1":
        t0 = time.perf_counter()
        canvas = a1_gray_lstar.read_canvas(dataset_a1_dir, stem)
        mask = npy_io.foreground_mask(canvas)
        timings["read_source_s"] = time.perf_counter() - t0
        return canvas, mask
    if member == "A2":
        t0 = time.perf_counter()
        canvas = a2_gray_standardized.read_canvas(dataset_a2_dir, stem)
        mask = npy_io.foreground_mask(canvas)
        timings["read_source_s"] = time.perf_counter() - t0
        return canvas, mask
    if member == "A7":
        t0 = time.perf_counter()
        canvas = a7_chroma_only.read_canvas(dataset_a7_dir, stem)
        mask = npy_io.foreground_mask(canvas)
        timings["read_source_s"] = time.perf_counter() - t0
        return canvas, mask

    t0 = time.perf_counter()
    canvas = npy_io.read_canvas(dataset_dir, stem)
    mask = npy_io.foreground_mask(canvas)
    timings["read_source_s"] = time.perf_counter() - t0

    if member == "A6":
        t0 = time.perf_counter()
        cached = _read_a6_caches(dataset_a1_dir, dataset_a6_class_dir, stem) if dataset_a6_class_dir else None

        if cached is not None:
            labels, grayscale = cached
            if fixed_a6_category is not None:
                canvas, _mask, a6_meta = a6_colour_removal.apply_from_cache(canvas, labels, grayscale, fixed_a6_category)
                meta["A6_category"] = fixed_a6_category
                meta["A6"] = a6_meta
            else:
                canvas, category, a6_meta = a6_colour_removal.draw_and_apply_from_cache(
                    canvas, mask, labels, grayscale, rng
                )
                meta["A6_category"] = category
                if category is not None:
                    meta["A6"] = a6_meta
        elif fixed_a6_category is not None:
            canvas, a6_meta = a6_colour_removal.apply(canvas, mask, fixed_a6_category)
            meta["A6_category"] = fixed_a6_category
            meta["A6"] = a6_meta
        else:
            # draw + apply share one classification pass instead of two -
            # see a6_colour_removal.draw_and_apply()'s docstring
            canvas, category, a6_meta = a6_colour_removal.draw_and_apply(canvas, mask, rng)
            meta["A6_category"] = category
            if category is not None:
                meta["A6"] = a6_meta
        timings["A6_s"] = time.perf_counter() - t0
    return canvas, mask


def _read_a6_caches(dataset_a1_dir: Path, dataset_a6_class_dir: Path, stem: str):
    """Both offline caches A6's fast path needs, or None if either is
    missing for this specimen - the caller falls back to the online
    classifier silently, so a partially-built cache never breaks a run."""
    try:
        labels = a6_colour_removal.read_classification_cache_entry(dataset_a6_class_dir, stem)
        grayscale = a1_gray_lstar.read_canvas(dataset_a1_dir, stem)
    except FileNotFoundError:
        return None
    return labels, grayscale
