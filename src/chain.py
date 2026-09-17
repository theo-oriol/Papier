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

    # --- step 2: vertical flip, shared by the whole sac - drawn here (same
    # rng draw order as before) but *applied* after step 3's median blur
    # instead of before it. A symmetric kernel's median blur commutes
    # exactly with a vertical flip, border replication included (verified
    # bit-identical on real-shaped data), so blurring the still-unflipped,
    # C-contiguous canvas first and flipping the result afterward gives the
    # same output while avoiding cv2.medianBlur's ~10% slowdown when handed
    # a flipped (negative-stride) view instead of a normal array. ---
    flip = bool(geometry_rng.random() < 0.5)
    meta["flip"] = flip

    # --- step 3: median blur ---
    t0 = time.perf_counter()
    canvas = cv2.medianBlur(canvas, MEDIAN_BLUR_K)
    timings["median_blur_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if flip:
        canvas = np.flipud(canvas)
        mask = np.flipud(mask)
    timings["flip_s"] = time.perf_counter() - t0

    # --- step 4: A4, image level ---
    if "A4" in condition:
        t0 = time.perf_counter()
        params = fixed_a4_params if fixed_a4_params is not None else a4_directional_blur.draw_params(rng)
        canvas = a4_directional_blur.apply(canvas, **params)
        meta["A4"] = params
        timings["A4_s"] = time.perf_counter() - t0

    # --- step 5: sample positions (from the *un-flipped* candidate table:
    # flipping y -> H-P-y is exact for a permutation, so we flip the table's
    # own coordinates instead of re-deriving candidates for the flipped
    # canvas) ---
    t0 = time.perf_counter()
    crop_specs = []  # (size, y, x)
    for size in CROP_SIZES:
        candidates = load_candidates(dataset_dir, stem, size)
        positions = sample_positions(candidates, N_CROPS_PER_SIZE, geometry_rng)
        if flip:
            canvas_h = canvas.shape[0]
            positions = [(canvas_h - size - y, x) for (y, x) in positions]
        crop_specs.extend((size, y, x) for y, x in positions)
    meta["positions"] = crop_specs
    timings["sample_positions_s"] = time.perf_counter() - t0

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
            target_alpha = a5_spectral_slope.target_alpha_for_crop_source(size)
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
                canvas, a6_meta = a6_colour_removal.apply_from_cache(canvas, labels, grayscale, fixed_a6_category)
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
