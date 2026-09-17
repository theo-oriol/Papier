"""Colour-space ablations and their scientific scope.

All colour operations are performed in CIE L*a*b* (D65), after conversion from
sRGB.  This makes luminance/chroma manipulations explicit and avoids treating
the numerical R, G and B display primaries as independent biological colours.

The nomenclature follows the 12 broad avian colour classes of Delhey et al.
(2023), *PNAS*, doi:10.1073/pnas.2217692120. Pixel assignment starts from the
published CN11 colour-name probabilities (van de Weijer et al. 2009). A small,
documented calibration layer moves only the boundaries that systematic QC
on colour-calibrated bird images showed to be problematic. The ordinary
``delhey_no_*`` variants use hard calibrated labels; ``delhey_soft_no_*`` use
the calibrated category memberships as graded masks. Only the eight chromatic
classes are ablated. Colour removal itself is a new experimental intervention:
a change in CNN error means that the removed class carried non-redundant
predictive information, not that the colour is a causal adaptation or a proven
mediator of sexual selection.

See ``README_plumage_ablations.md`` for the planned paired estimands and the
distinction between moderation and causal mediation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.io import loadmat
from skimage import color


DELHEY_CATEGORIES: Tuple[str, ...] = (
    "blue",
    "purple",
    "red",
    "yellow",
    "green",
    "rufous",
    "light_brown",
    "dark_brown",
    "light_gray",
    "dark_gray",
    "black",
    "white",
)

DELHEY_CHROMATIC_CATEGORIES: Tuple[str, ...] = (
    "blue",
    "purple",
    "red",
    "yellow",
    "green",
    "rufous",
    "light_brown",
    "dark_brown",
)
DELHEY_CHROMATIC = frozenset(DELHEY_CHROMATIC_CATEGORIES)

CN11_CATEGORIES: Tuple[str, ...] = (
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
)
CN11_RESOURCE_PATH = Path(__file__).resolve().parent / "resources" / "w2c.mat"
CN11_SOURCE_URL = (
    "https://lear.inrialpes.fr/people/vandeweijer/code/ColorNaming.tar"
)

# CN11 remains the baseline. These CIELCh conditions are local calibration
# gates, not a replacement partition of colour space. CIE hue angles are not
# HSV/HSL angles: representative sRGB red, rufous, yellow, green, blue and
# purple occur near 40, 54, 91, 136, 306 and 328 degrees. Every constant is
# written to the run manifest.
DELHEY_CN11_CALIBRATION = {
    "brown_light_split_L": 42.0,
    "gray_light_split_L": 42.0,
    "black_to_dark_gray_lightness_min": 25.0,
    "brown_to_gray_chroma_max": 12.0,
    "brown_to_gray_hue_min": 60.0,
    "warm_light_gray_lightness_min": 42.0,
    "warm_light_gray_chroma_max": 20.0,
    "warm_light_gray_hue_min": 72.0,
    "warm_light_gray_cn11_ratio_min": 0.30,
    "pink_red_hue_max": 45.0,
    "pink_red_hue_min_wrapped": 330.0,
    "blue_donor_chroma_min": 5.0,
    "blue_hue_min": 170.0,
    "blue_hue_max": 315.0,
    "red_core_hue_max": 41.0,
    "red_core_chroma_min": 12.0,
    "red_orange_hue_max": 47.0,
    "red_orange_chroma_min": 30.0,
    "red_wrapped_hue_min": 350.0,
    "yellow_hue_min": 70.0,
    "yellow_hue_max": 100.0,
    "yellow_chroma_min": 30.0,
    "rufous_hue_min": 41.0,
    "rufous_hue_max": 70.0,
    "rufous_chroma_min": 22.0,
    "rufous_dark_lightness_max": 30.0,
    "rufous_dark_chroma_max": 35.0,
    "rufous_pale_lightness_min": 45.0,
    "rufous_pale_chroma_max": 45.0,
    "rufous_to_light_brown_chroma_max": 30.0,
    "rufous_yellow_edge_hue_max": 82.0,
    "rufous_yellow_edge_chroma_min": 22.0,
    "rufous_yellow_edge_chroma_max": 30.0,
    "rufous_yellow_edge_lightness_max": 45.0,
    # Warm off-white plumage is often called brown by CN11. The allowed chroma
    # increases gently with lightness, reflecting a weak cream cast, but is
    # capped so genuinely tan feathers remain light brown.
    "cream_white_lightness_min": 58.0,
    "cream_white_chroma_at_L50": 14.0,
    "cream_white_chroma_per_L": 0.35,
    "cream_white_chroma_max": 24.0,
    # Olive plumage is sometimes named brown/orange/yellow by CN11. Rescue it
    # only where CN11 itself assigns appreciable green probability relative to
    # every competing warm name. A slightly looser ratio is permitted in dark
    # plumage, where all colour-name probabilities are flatter.
    "green_rescue_hue_min": 65.0,
    "green_rescue_dark_hue_min": 72.0,
    "green_rescue_dark_hue_max": 85.0,
    "green_rescue_main_hue_min": 85.0,
    "green_rescue_hue_max": 145.0,
    "green_rescue_chroma_min": 10.0,
    "green_rescue_probability_min": 0.08,
    "green_rescue_warm_ratio_min": 0.25,
    # The low-hue olive tail overlaps strongly with warm yellow-brown plumage.
    # These bounds are therefore a compromise: they recover dark olive backs
    # in Piculus and Saltator without restoring the broad v4 green expansion.
    "green_rescue_dark_ratio_min": 0.25,
    "green_rescue_dark_lightness_max": 34.0,
    "green_rescue_dark_chroma_max": 26.0,
    "green_rescue_vivid_yellow_guard_hue_max": 80.0,
    "green_rescue_vivid_yellow_guard_chroma_min": 35.0,
    # Dark blue CN11 pixels can be won by black or grey because luminance is
    # low. Hue, chroma and a non-negligible CN11 blue probability are all
    # required, so neutral black is not relabelled.
    "dark_blue_hue_min": 230.0,
    "dark_blue_hue_max": 330.0,
    "dark_blue_chroma_min": 5.0,
    "dark_blue_probability_min": 0.06,
    # Further local corrections for pale yellow-white, dark maroon-purple and
    # the lowest-lightness brown/grey tail.
    "pale_yellow_white_lightness_min": 85.0,
    "pale_yellow_white_chroma_max": 25.0,
    "dark_red_purple_lightness_max": 22.0,
    "dark_red_purple_chroma_max": 20.0,
    "dark_red_purple_cn11_ratio_min": 0.60,
    "dark_mark_black_lightness_max": 22.0,
    # Soft calibration only: logistic transition widths and maximum fractions
    # of donor membership transferred to the neighbouring category.
    "soft_lightness_width": 4.0,
    "soft_hue_width": 4.0,
    "soft_chroma_width": 4.0,
    "soft_blue_transfer_max": 0.85,
    "soft_rufous_transfer_max": 0.90,
    "soft_rufous_to_light_brown_transfer_max": 0.90,
    "soft_red_transfer_max": 0.85,
    "soft_yellow_transfer_max": 0.85,
    "soft_cream_white_transfer_max": 0.95,
    "soft_black_gray_transfer_max": 0.95,
    "soft_brown_gray_transfer_max": 0.95,
    "soft_green_transfer_max": 0.90,
    "soft_dark_blue_transfer_max": 0.90,
    "soft_pale_yellow_white_transfer_max": 0.95,
    "soft_dark_red_purple_transfer_max": 0.90,
    "soft_dark_mark_black_transfer_max": 0.95,
}


@lru_cache(maxsize=1)
def _load_cn11_table() -> np.ndarray:
    """Load the published 32^3 sRGB-to-CN11 probability lookup table."""

    if not CN11_RESOURCE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing CN11 lookup table: {CN11_RESOURCE_PATH}"
        )
    table = np.asarray(loadmat(CN11_RESOURCE_PATH)["w2c"], dtype=np.float64)
    if table.shape != (32768, 11):
        raise ValueError(f"Unexpected CN11 table shape: {table.shape}")
    if not np.all(np.isfinite(table)) or np.any(table < 0):
        raise ValueError("CN11 table contains invalid probabilities")
    table.setflags(write=False)
    return table


def _check_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 RGB array, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("RGB image contains non-finite values")
    if arr.min() < -1e-8 or arr.max() > 1.0 + 1e-8:
        raise ValueError("RGB input must be scaled to [0, 1]")
    return np.clip(arr, 0.0, 1.0)


def _check_foreground(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"Foreground mask shape {mask.shape} != image shape {shape}")
    if not np.any(mask):
        raise ValueError("Foreground mask is empty")
    return mask


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB image in [0, 1] to CIE Lab under D65."""

    return color.rgb2lab(_check_rgb(rgb), illuminant="D65")


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE Lab under D65 to clipped sRGB in [0, 1]."""

    lab = np.asarray(lab, dtype=np.float64)
    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 Lab array, got {lab.shape}")
    with np.errstate(invalid="ignore"):
        rgb = color.lab2rgb(lab, illuminant="D65")
    return np.clip(rgb, 0.0, 1.0)


def achromatic_raw(
    rgb: np.ndarray, foreground: Optional[np.ndarray] = None
) -> np.ndarray:
    """Remove all chroma while retaining each foreground pixel's L*."""

    arr = _check_rgb(rgb)
    lab = rgb_to_lab(arr)
    fg = _check_foreground(foreground, lab.shape[:2])
    lab[..., 1][fg] = 0.0
    lab[..., 2][fg] = 0.0
    result = lab_to_rgb(lab)
    result[~fg] = arr[~fg]
    return result


def achromatic_equalized(
    rgb: np.ndarray,
    foreground: Optional[np.ndarray] = None,
    target_mean_l: float = 50.0,
    target_sd_l: float = 15.0,
    epsilon: float = 1e-8,
    reference_mean_l: Optional[float] = None,
    reference_sd_l: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Remove chroma and standardise global foreground lightness and contrast.

    This is a transparent global mean/RMS-contrast equalisation, not CLAHE.  It
    therefore keeps the spatial organisation of achromatic pattern intact.
    """

    if target_sd_l <= 0:
        raise ValueError("target_sd_l must be positive")
    arr = _check_rgb(rgb)
    lab = rgb_to_lab(arr)
    fg = _check_foreground(foreground, lab.shape[:2])
    values = lab[..., 0][fg]

    # WHICH statistics define "global"?  Supplying reference_mean_l/sd measured
    # on the WHOLE BIRD removes between-bird lightness and contrast while
    # keeping the lightness differences BETWEEN crops of one bird -- the
    # dorsal/ventral gradient among them, which is plumage biology rather than
    # a nuisance.  Falling back to the crop's own statistics removes that
    # variation too, and then answers a different question.
    if reference_mean_l is None or reference_sd_l is None:
        old_mean = float(values.mean())
        old_sd = float(values.std(ddof=0))
        reference_scope = "crop"
    else:
        old_mean = float(reference_mean_l)
        old_sd = float(reference_sd_l)
        reference_scope = "whole_image"
    if old_sd <= epsilon:
        scaled = np.full(values.shape, target_mean_l, dtype=np.float64)
    else:
        scaled = (values - old_mean) * (target_sd_l / old_sd) + target_mean_l
    lab[..., 0][fg] = np.clip(scaled, 0.0, 100.0)
    lab[..., 1][fg] = 0.0
    lab[..., 2][fg] = 0.0
    result = lab_to_rgb(lab)
    result[~fg] = arr[~fg]
    out_l = rgb_to_lab(result)[..., 0][fg]
    metadata = {
        "equalisation_reference_scope": reference_scope,
        "reference_mean_L": old_mean,
        "reference_sd_L": old_sd,
        "crop_mean_L": float(values.mean()),
        "crop_sd_L": float(values.std(ddof=0)),
        "original_mean_L": old_mean,
        "original_sd_L": old_sd,
        "target_mean_L": float(target_mean_l),
        "target_sd_L": float(target_sd_l),
        "realized_mean_L": float(out_l.mean()),
        "realized_sd_L": float(out_l.std(ddof=0)),
    }
    return result, metadata


def chroma_isoluminant(
    rgb: np.ndarray,
    foreground: Optional[np.ndarray] = None,
    luminance_l: float = 50.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Set foreground L* to one value and audit realised chroma after sRGB conversion.

    Constant-L* Lab colours can lie outside the sRGB gamut.  Conversion then
    reduces chroma in a hue-dependent way.  The returned metadata quantifies
    that loss overall and within each Delhey category; callers must not describe
    this arm as perfectly chroma-preserving without consulting those fields.
    """

    if not 0.0 <= luminance_l <= 100.0:
        raise ValueError("luminance_l must lie in [0, 100]")
    arr = _check_rgb(rgb)
    lab = rgb_to_lab(arr)
    fg = _check_foreground(foreground, lab.shape[:2])
    intended = lab.copy()
    intended[..., 0][fg] = luminance_l
    output = lab_to_rgb(intended)
    output[~fg] = arr[~fg]
    realised = rgb_to_lab(output)

    intended_chroma = np.hypot(intended[..., 1], intended[..., 2])
    realised_chroma = np.hypot(realised[..., 1], realised[..., 2])
    chromatic = fg & (intended_chroma > 1.0)
    retention = np.divide(
        realised_chroma,
        intended_chroma,
        out=np.ones_like(realised_chroma),
        where=intended_chroma > 1.0,
    )
    delta_e = np.linalg.norm(realised - intended, axis=-1)

    labels = classify_delhey_rgb(rgb, fg)
    retention_by_category: Dict[str, Dict[str, object]] = {}
    for category_index, category in enumerate(DELHEY_CATEGORIES):
        selected = fg & (labels == category_index)
        selected_chromatic = selected & chromatic
        retention_by_category[category] = {
            "n_pixels": int(selected.sum()),
            "mean_chroma_retention_ratio": (
                float(retention[selected_chromatic].mean())
                if np.any(selected_chromatic)
                else None
            ),
            "mean_delta_E_roundtrip": (
                float(delta_e[selected].mean()) if np.any(selected) else None
            ),
        }

    metadata: Dict[str, object] = {
        "mode": "constant_L_preserve_ab_before_sRGB_gamut_mapping",
        "constant_L": float(luminance_l),
        "n_foreground_pixels": int(fg.sum()),
        "n_chromatic_pixels": int(chromatic.sum()),
        "mean_intended_chroma": float(intended_chroma[fg].mean()),
        "mean_realized_chroma": float(realised_chroma[fg].mean()),
        "mean_chroma_retention_ratio": (
            float(retention[chromatic].mean()) if np.any(chromatic) else None
        ),
        "median_chroma_retention_ratio": (
            float(np.median(retention[chromatic])) if np.any(chromatic) else None
        ),
        "fraction_chromatic_pixels_losing_gt_5pct_chroma": (
            float(np.mean(retention[chromatic] < 0.95)) if np.any(chromatic) else 0.0
        ),
        "fraction_chromatic_pixels_losing_gt_20pct_chroma": (
            float(np.mean(retention[chromatic] < 0.80)) if np.any(chromatic) else 0.0
        ),
        "mean_delta_E_roundtrip": float(delta_e[fg].mean()),
        "fraction_pixels_delta_E_gt_1": float(np.mean(delta_e[fg] > 1.0)),
        "gamut_or_roundtrip_distortion_detected": bool(np.any(delta_e[fg] > 1.0)),
        "chroma_retention_by_delhey_category": retention_by_category,
    }
    return output, metadata


def cn11_probabilities(
    rgb: np.ndarray,
    foreground: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the published probabilities for the 11 basic CN11 terms.

    CN11 quantises each 8-bit RGB channel into 32 bins. Background scores are
    zero. The probabilities are not merged before classification: doing so
    would give a target class such as red+pink a mechanical advantage over
    source classes represented by a single CN11 term.
    """

    arr = _check_rgb(rgb)
    fg = _check_foreground(foreground, arr.shape[:2])
    rgb8 = np.rint(arr[fg] * 255.0).astype(np.uint8)
    index = (
        rgb8[:, 0].astype(np.int32) // 8
        + 32 * (rgb8[:, 1].astype(np.int32) // 8)
        + 1024 * (rgb8[:, 2].astype(np.int32) // 8)
    )
    probabilities = np.zeros((*arr.shape[:2], len(CN11_CATEGORIES)))
    probabilities[fg] = _load_cn11_table()[index]
    return probabilities


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """Numerically stable logistic gate used by fuzzy boundary calibration.

    sigmoid(x) = exp(-log(1+exp(-x))) = exp(-logaddexp(0, -x)) - the same
    numerically stable formula as before, but branchless: the previous
    version's boolean masking (value[positive] = ..., value[~positive] = ...)
    allocates and gather/scatters twice per call, which measurably adds up
    over the ~50 calls one classify_delhey_rgb() makes on a multi-million
    pixel image (profiled at ~4s of a ~13s total)."""

    value = np.asarray(value, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -value))


def _window_gate(
    value: np.ndarray, lower: float, upper: float, width: float
) -> np.ndarray:
    return _sigmoid((value - lower) / width) * _sigmoid((upper - value) / width)


def _transfer_membership(
    memberships: np.ndarray,
    donors: Tuple[int, ...],
    recipient: int,
    fraction: np.ndarray,
) -> None:
    """Move, rather than duplicate, probability mass between local neighbours."""

    fraction = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0)
    for donor in donors:
        moved = memberships[..., donor] * fraction
        memberships[..., donor] -= moved
        memberships[..., recipient] += moved


def _base_cn11_labels_and_memberships(
    probabilities: np.ndarray,
    raw_winner: np.ndarray,
    lightness: np.ndarray,
    hue: np.ndarray,
    foreground: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map CN11 to Delhey names without changing the baseline boundaries."""

    shape = raw_winner.shape
    labels = np.full(shape, -1, dtype=np.int16)
    memberships = np.zeros((*shape, len(DELHEY_CATEGORIES)), dtype=np.float64)
    cn = {name: idx for idx, name in enumerate(CN11_CATEGORIES)}
    target = {name: idx for idx, name in enumerate(DELHEY_CATEGORIES)}
    rule = DELHEY_CN11_CALIBRATION

    direct = {
        "black": "black",
        "blue": "blue",
        "green": "green",
        "orange": "rufous",
        "purple": "purple",
        "red": "red",
        "white": "white",
        "yellow": "yellow",
    }
    for source_name, target_name in direct.items():
        labels[foreground & (raw_winner == cn[source_name])] = target[target_name]
        memberships[..., target[target_name]] += probabilities[..., cn[source_name]]

    brown_light_hard = lightness >= rule["brown_light_split_L"]
    brown_light_soft = _sigmoid(
        (lightness - rule["brown_light_split_L"])
        / rule["soft_lightness_width"]
    )
    gray_light_hard = lightness >= rule["gray_light_split_L"]
    gray_light_soft = _sigmoid(
        (lightness - rule["gray_light_split_L"])
        / rule["soft_lightness_width"]
    )
    for source_name, target_stem, light_hard, light_soft in (
        ("brown", "brown", brown_light_hard, brown_light_soft),
        ("gray", "gray", gray_light_hard, gray_light_soft),
    ):
        source = cn[source_name]
        source_hard = foreground & (raw_winner == source)
        labels[source_hard & light_hard] = target[f"light_{target_stem}"]
        labels[source_hard & ~light_hard] = target[f"dark_{target_stem}"]
        memberships[..., target[f"light_{target_stem}"]] += (
            probabilities[..., source] * light_soft
        )
        memberships[..., target[f"dark_{target_stem}"]] += (
            probabilities[..., source] * (1.0 - light_soft)
        )

    # CN11 has no beige/tan term. Split its pink membership by hue so warm
    # beige does not become red, while genuinely pink/magenta plumage remains
    # part of Delhey's broad red class.
    pink_hard = foreground & (raw_winner == cn["pink"])
    pink_is_red_hard = pink_hard & (
        (hue <= rule["pink_red_hue_max"])
        | (hue >= rule["pink_red_hue_min_wrapped"])
    )
    labels[pink_is_red_hard] = target["red"]
    pink_is_brown_hard = pink_hard & ~pink_is_red_hard
    labels[pink_is_brown_hard & brown_light_hard] = target["light_brown"]
    labels[pink_is_brown_hard & ~brown_light_hard] = target["dark_brown"]

    width = rule["soft_hue_width"]
    pink_red_gate = np.where(
        hue < 180.0,
        _sigmoid((rule["pink_red_hue_max"] - hue) / width),
        _sigmoid((hue - rule["pink_red_hue_min_wrapped"]) / width),
    )
    pink_probability = probabilities[..., cn["pink"]]
    memberships[..., target["red"]] += pink_probability * pink_red_gate
    pink_brown = pink_probability * (1.0 - pink_red_gate)
    memberships[..., target["light_brown"]] += pink_brown * brown_light_soft
    memberships[..., target["dark_brown"]] += pink_brown * (1.0 - brown_light_soft)
    return labels, memberships


def _classify_delhey_rgb_details_impl(
    rgb: np.ndarray,
    foreground: Optional[np.ndarray] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return locally calibrated CN11 hard labels and fuzzy memberships."""

    arr = _check_rgb(rgb)
    fg = _check_foreground(foreground, arr.shape[:2])
    probabilities = cn11_probabilities(arr, fg)
    raw_winner = np.full(arr.shape[:2], -1, dtype=np.int16)
    raw_winner[fg] = np.argmax(probabilities[fg], axis=1).astype(np.int16)
    lab = rgb_to_lab(arr)
    lightness = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.mod(np.degrees(np.arctan2(lab[..., 2], lab[..., 1])), 360.0)
    labels, memberships = _base_cn11_labels_and_memberships(
        probabilities, raw_winner, lightness, hue, fg
    )

    cn = {name: idx for idx, name in enumerate(CN11_CATEGORIES)}
    target = {name: idx for idx, name in enumerate(DELHEY_CATEGORIES)}
    rule = DELHEY_CN11_CALIBRATION

    # HARD primary: preserve the CN11 label except at predeclared source
    # boundaries. Conditions are only allowed to override plausible donor
    # classes, preventing a global repartition of colour space.
    blue_override = (
        fg
        & np.isin(raw_winner, (cn["green"], cn["gray"]))
        & (chroma >= rule["blue_donor_chroma_min"])
        & (hue >= rule["blue_hue_min"])
        & (hue < rule["blue_hue_max"])
    )
    labels[blue_override] = target["blue"]

    red_override = (
        fg
        & np.isin(raw_winner, (cn["brown"], cn["orange"], cn["pink"]))
        & (
            (
                ((hue < rule["red_core_hue_max"]) | (hue >= rule["red_wrapped_hue_min"]))
                & (chroma >= rule["red_core_chroma_min"])
            )
            | (
                (hue >= rule["red_core_hue_max"])
                & (hue < rule["red_orange_hue_max"])
                & (chroma >= rule["red_orange_chroma_min"])
            )
        )
    )
    labels[red_override] = target["red"]

    yellow_override = (
        fg
        & np.isin(raw_winner, (cn["brown"], cn["orange"], cn["green"]))
        & (hue >= rule["yellow_hue_min"])
        & (hue < rule["yellow_hue_max"])
        & (chroma >= rule["yellow_chroma_min"])
    )
    labels[yellow_override] = target["yellow"]

    rufous_override = (
        fg
        & np.isin(raw_winner, (cn["brown"], cn["orange"]))
        & (hue >= rule["rufous_hue_min"])
        & (hue < rule["rufous_hue_max"])
        & (chroma >= rule["rufous_chroma_min"])
        & ~(
            (lightness < rule["rufous_dark_lightness_max"])
            & (chroma < rule["rufous_dark_chroma_max"])
        )
        & ~(
            (lightness >= rule["rufous_pale_lightness_min"])
            & (chroma < rule["rufous_pale_chroma_max"])
        )
    )
    rufous_override |= (
        fg
        & np.isin(raw_winner, (cn["brown"], cn["orange"]))
        & (hue >= rule["yellow_hue_min"])
        & (hue < rule["rufous_yellow_edge_hue_max"])
        & (chroma >= rule["rufous_yellow_edge_chroma_min"])
        & (chroma < rule["rufous_yellow_edge_chroma_max"])
        & (lightness < rule["rufous_yellow_edge_lightness_max"])
    )
    labels[rufous_override & ~red_override & ~yellow_override] = target["rufous"]

    # CN11 orange maps directly to rufous before the override above. Its pale,
    # weakly chromatic tail is visually tan rather than rufous and therefore
    # belongs with light brown. This is deliberately narrower than the gate
    # that merely prevents brown from being promoted to rufous.
    pale_rufous_to_light_brown_override = (
        fg
        & (labels == target["rufous"])
        & (lightness >= rule["rufous_pale_lightness_min"])
        & (chroma <= rule["rufous_to_light_brown_chroma_max"])
    )
    labels[pale_rufous_to_light_brown_override] = target["light_brown"]

    # Correct the systematic olive-green failure without treating hue alone as
    # sufficient evidence. The CN11 green score must be appreciable and close
    # to the strongest warm alternative. This retains true yellows in Serinus,
    # Sturnella and other calibration birds while rescuing olive woodpeckers,
    # parrots, pigeons and tanagers.
    warm_probability = np.maximum.reduce(
        (
            probabilities[..., cn["brown"]],
            probabilities[..., cn["orange"]],
            probabilities[..., cn["yellow"]],
        )
    )
    green_to_warm_ratio = probabilities[..., cn["green"]] / np.maximum(
        warm_probability, 1e-12
    )
    green_rescue_override = (
        fg
        & np.isin(
            labels,
            (
                target["yellow"],
                target["rufous"],
                target["light_brown"],
                target["dark_brown"],
                target["light_gray"],
                target["dark_gray"],
            ),
        )
        & (probabilities[..., cn["green"]] >= rule["green_rescue_probability_min"])
        & (chroma >= rule["green_rescue_chroma_min"])
        & (hue >= rule["green_rescue_hue_min"])
        & (hue < rule["green_rescue_hue_max"])
        & (
            (
                (hue >= rule["green_rescue_main_hue_min"])
                & (green_to_warm_ratio >= rule["green_rescue_warm_ratio_min"])
            )
            | (
                (lightness <= rule["green_rescue_dark_lightness_max"])
                & (hue >= rule["green_rescue_dark_hue_min"])
                & (hue < rule["green_rescue_dark_hue_max"])
                & (chroma <= rule["green_rescue_dark_chroma_max"])
                & (raw_winner == cn["brown"])
                & (green_to_warm_ratio >= rule["green_rescue_dark_ratio_min"])
            )
        )
        & ~(
            (hue < rule["green_rescue_vivid_yellow_guard_hue_max"])
            & (chroma >= rule["green_rescue_vivid_yellow_guard_chroma_min"])
        )
    )
    labels[green_rescue_override] = target["green"]

    pale_yellow_white_override = (
        fg
        & (labels == target["yellow"])
        & (lightness >= rule["pale_yellow_white_lightness_min"])
        & (chroma <= rule["pale_yellow_white_chroma_max"])
    )
    labels[pale_yellow_white_override] = target["white"]

    # CN11 frequently calls warm off-white feathers brown. Only pixels that are
    # still light brown after the chromatic overrides may become white. The
    # sloped boundary admits a slightly stronger cream cast at high L* while
    # retaining genuinely tan, more chromatic feathers as light brown.
    cream_white_chroma_limit = np.minimum(
        rule["cream_white_chroma_max"],
        rule["cream_white_chroma_at_L50"]
        + rule["cream_white_chroma_per_L"] * (lightness - 50.0),
    )
    cream_white_override = (
        fg
        & (labels == target["light_brown"])
        & (lightness >= rule["cream_white_lightness_min"])
        & (chroma <= cream_white_chroma_limit)
    )
    labels[cream_white_override] = target["white"]

    # Very weak chroma is more consistent with grey than brown, even when CN11
    # narrowly favours its brown name. This corrects warm-grey feather pixels
    # without moving clearly chromatic brown or rufous plumage. The light/dark
    # grey boundary is intentionally independent of the brown boundary.
    low_chroma_brown_override = (
        fg
        & np.isin(labels, (target["light_brown"], target["dark_brown"]))
        & (chroma <= rule["brown_to_gray_chroma_max"])
        & (hue >= rule["brown_to_gray_hue_min"])
    )
    gray_light_hard = lightness >= rule["gray_light_split_L"]
    labels[low_chroma_brown_override & gray_light_hard] = target["light_gray"]
    labels[low_chroma_brown_override & ~gray_light_hard] = target["dark_gray"]

    # A second, deliberately narrow warm-grey gate catches pale, weakly
    # chromatic belly feathers where CN11 only narrowly favours brown. Requiring
    # both a yellowish hue and a substantial grey:brown probability ratio
    # avoids turning ordinary tan plumage into grey.
    gray_to_brown_ratio = probabilities[..., cn["gray"]] / np.maximum(
        probabilities[..., cn["brown"]], 1e-12
    )
    warm_light_gray_override = (
        fg
        & np.isin(labels, (target["light_brown"], target["dark_brown"]))
        & (lightness >= rule["warm_light_gray_lightness_min"])
        & (chroma <= rule["warm_light_gray_chroma_max"])
        & (hue >= rule["warm_light_gray_hue_min"])
        & (gray_to_brown_ratio >= rule["warm_light_gray_cn11_ratio_min"])
    )
    labels[warm_light_gray_override] = target["light_gray"]

    # CN11 black includes the upper end of near-black. Retain truly dark pixels
    # as black but call the lighter tail of that distribution dark grey.
    black_to_dark_gray_override = (
        fg
        & (labels == target["black"])
        & (lightness >= rule["black_to_dark_gray_lightness_min"])
    )
    labels[black_to_dark_gray_override] = target["dark_gray"]

    dark_blue_override = (
        fg
        & np.isin(labels, (target["black"], target["dark_gray"]))
        & (probabilities[..., cn["blue"]] >= rule["dark_blue_probability_min"])
        & (chroma >= rule["dark_blue_chroma_min"])
        & (hue >= rule["dark_blue_hue_min"])
        & (hue < rule["dark_blue_hue_max"])
    )
    labels[dark_blue_override] = target["blue"]

    purple_to_red_ratio = probabilities[..., cn["purple"]] / np.maximum(
        probabilities[..., cn["red"]], 1e-12
    )
    dark_red_purple_override = (
        fg
        & (labels == target["red"])
        & (lightness <= rule["dark_red_purple_lightness_max"])
        & (chroma <= rule["dark_red_purple_chroma_max"])
        & (purple_to_red_ratio >= rule["dark_red_purple_cn11_ratio_min"])
    )
    labels[dark_red_purple_override] = target["purple"]

    dark_mark_black_override = (
        fg
        & np.isin(
            labels,
            (
                target["light_brown"],
                target["dark_brown"],
                target["light_gray"],
                target["dark_gray"],
            ),
        )
        & (lightness <= rule["dark_mark_black_lightness_max"])
    )
    labels[dark_mark_black_override] = target["black"]

    # FUZZY sensitivity: smoothly transfer probability mass across exactly the
    # same local boundaries. Every transfer is conservative, so memberships
    # remain non-negative and sum to one on foreground pixels.
    hue_width = rule["soft_hue_width"]
    chroma_width = rule["soft_chroma_width"]
    gray_light_soft = _sigmoid(
        (lightness - rule["gray_light_split_L"])
        / rule["soft_lightness_width"]
    )
    blue_gate = (
        _window_gate(hue, rule["blue_hue_min"], rule["blue_hue_max"], hue_width)
        * _sigmoid((chroma - rule["blue_donor_chroma_min"]) / chroma_width)
        * rule["soft_blue_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["green"], target["light_gray"], target["dark_gray"]),
        target["blue"],
        blue_gate,
    )

    rufous_gate = (
        _window_gate(hue, rule["rufous_hue_min"], rule["rufous_yellow_edge_hue_max"], hue_width)
        * _sigmoid((chroma - rule["rufous_chroma_min"]) / chroma_width)
        * _sigmoid((lightness - 25.0) / rule["soft_lightness_width"])
        * _sigmoid(
            (rule["rufous_pale_lightness_min"] - lightness)
            / rule["soft_lightness_width"]
        )
        * rule["soft_rufous_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["light_brown"], target["dark_brown"]),
        target["rufous"],
        rufous_gate,
    )

    pale_rufous_to_light_brown_gate = (
        _sigmoid(
            (lightness - rule["rufous_pale_lightness_min"])
            / rule["soft_lightness_width"]
        )
        * _sigmoid(
            (rule["rufous_to_light_brown_chroma_max"] - chroma) / chroma_width
        )
        * rule["soft_rufous_to_light_brown_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["rufous"],),
        target["light_brown"],
        pale_rufous_to_light_brown_gate,
    )

    red_hue_gate = np.maximum(
        _sigmoid((rule["red_core_hue_max"] - hue) / hue_width)
        * (hue < 180.0),
        _sigmoid((hue - rule["red_wrapped_hue_min"]) / hue_width)
        * (hue >= 180.0),
    )
    red_gate = (
        red_hue_gate
        * _sigmoid((chroma - rule["red_core_chroma_min"]) / chroma_width)
        * rule["soft_red_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["rufous"], target["light_brown"], target["dark_brown"]),
        target["red"],
        red_gate,
    )

    yellow_gate = (
        _window_gate(hue, rule["yellow_hue_min"], rule["yellow_hue_max"], hue_width)
        * _sigmoid((chroma - rule["yellow_chroma_min"]) / chroma_width)
        * rule["soft_yellow_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["rufous"], target["light_brown"], target["dark_brown"], target["green"]),
        target["yellow"],
        yellow_gate,
    )

    green_ratio_gate = np.maximum(
        _sigmoid(
            (green_to_warm_ratio - rule["green_rescue_warm_ratio_min"]) / 0.04
        )
        * _sigmoid(
            (hue - rule["green_rescue_main_hue_min"]) / hue_width
        ),
        _sigmoid(
            (green_to_warm_ratio - rule["green_rescue_dark_ratio_min"]) / 0.04
        )
        * _sigmoid(
            (hue - rule["green_rescue_dark_hue_min"]) / hue_width
        )
        * _sigmoid(
            (rule["green_rescue_dark_hue_max"] - hue) / hue_width
        )
        * _sigmoid(
            (rule["green_rescue_dark_lightness_max"] - lightness)
            / rule["soft_lightness_width"]
        ),
    )
    green_gate = (
        _window_gate(
            hue,
            rule["green_rescue_hue_min"],
            rule["green_rescue_hue_max"],
            hue_width,
        )
        * _sigmoid(
            (chroma - rule["green_rescue_chroma_min"]) / chroma_width
        )
        * np.where(
            hue < rule["green_rescue_dark_hue_max"],
            _sigmoid(
                (rule["green_rescue_dark_chroma_max"] - chroma) / chroma_width
            )
            * (raw_winner == cn["brown"]),
            1.0,
        )
        * _sigmoid(
            (probabilities[..., cn["green"]] - rule["green_rescue_probability_min"])
            / 0.02
        )
        * green_ratio_gate
        * (
            1.0
            - _sigmoid(
                (
                    rule["green_rescue_vivid_yellow_guard_hue_max"]
                    - hue
                )
                / hue_width
            )
            * _sigmoid(
                (
                    chroma
                    - rule["green_rescue_vivid_yellow_guard_chroma_min"]
                )
                / chroma_width
            )
        )
        * rule["soft_green_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (
            target["yellow"],
            target["rufous"],
            target["light_brown"],
            target["dark_brown"],
            target["light_gray"],
            target["dark_gray"],
        ),
        target["green"],
        green_gate,
    )

    pale_yellow_white_gate = (
        _sigmoid(
            (lightness - rule["pale_yellow_white_lightness_min"])
            / rule["soft_lightness_width"]
        )
        * _sigmoid(
            (rule["pale_yellow_white_chroma_max"] - chroma) / chroma_width
        )
        * rule["soft_pale_yellow_white_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["yellow"],),
        target["white"],
        pale_yellow_white_gate,
    )

    cream_white_gate = (
        _sigmoid(
            (lightness - rule["cream_white_lightness_min"])
            / rule["soft_lightness_width"]
        )
        * _sigmoid((cream_white_chroma_limit - chroma) / chroma_width)
        * rule["soft_cream_white_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["light_brown"],),
        target["white"],
        cream_white_gate,
    )

    brown_gray_gate = (
        _sigmoid(
            (rule["brown_to_gray_chroma_max"] - chroma)
            / rule["soft_chroma_width"]
        )
        * _sigmoid(
            (hue - rule["brown_to_gray_hue_min"]) / rule["soft_hue_width"]
        )
        * rule["soft_brown_gray_transfer_max"]
    )
    for donor in (target["light_brown"], target["dark_brown"]):
        moved = memberships[..., donor] * brown_gray_gate
        memberships[..., donor] -= moved
        memberships[..., target["light_gray"]] += moved * gray_light_soft
        memberships[..., target["dark_gray"]] += moved * (1.0 - gray_light_soft)

    warm_light_gray_gate = (
        _sigmoid(
            (lightness - rule["warm_light_gray_lightness_min"])
            / rule["soft_lightness_width"]
        )
        * _sigmoid(
            (rule["warm_light_gray_chroma_max"] - chroma) / chroma_width
        )
        * _sigmoid(
            (hue - rule["warm_light_gray_hue_min"]) / hue_width
        )
        * _sigmoid(
            (gray_to_brown_ratio - rule["warm_light_gray_cn11_ratio_min"])
            / 0.08
        )
        * rule["soft_brown_gray_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["light_brown"], target["dark_brown"]),
        target["light_gray"],
        warm_light_gray_gate,
    )

    black_gray_gate = (
        _sigmoid(
            (lightness - rule["black_to_dark_gray_lightness_min"])
            / rule["soft_lightness_width"]
        )
        * rule["soft_black_gray_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["black"],),
        target["dark_gray"],
        black_gray_gate,
    )

    dark_blue_gate = (
        _window_gate(
            hue,
            rule["dark_blue_hue_min"],
            rule["dark_blue_hue_max"],
            hue_width,
        )
        * _sigmoid((chroma - rule["dark_blue_chroma_min"]) / chroma_width)
        * _sigmoid(
            (probabilities[..., cn["blue"]] - rule["dark_blue_probability_min"])
            / 0.02
        )
        * rule["soft_dark_blue_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["black"], target["dark_gray"]),
        target["blue"],
        dark_blue_gate,
    )

    dark_red_purple_gate = (
        _sigmoid(
            (rule["dark_red_purple_lightness_max"] - lightness)
            / rule["soft_lightness_width"]
        )
        * _sigmoid(
            (rule["dark_red_purple_chroma_max"] - chroma) / chroma_width
        )
        * _sigmoid(
            (purple_to_red_ratio - rule["dark_red_purple_cn11_ratio_min"])
            / 0.10
        )
        * rule["soft_dark_red_purple_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (target["red"],),
        target["purple"],
        dark_red_purple_gate,
    )

    dark_mark_black_gate = (
        _sigmoid(
            (rule["dark_mark_black_lightness_max"] - lightness)
            / rule["soft_lightness_width"]
        )
        * rule["soft_dark_mark_black_transfer_max"]
    )
    _transfer_membership(
        memberships,
        (
            target["light_brown"],
            target["dark_brown"],
            target["light_gray"],
            target["dark_gray"],
        ),
        target["black"],
        dark_mark_black_gate,
    )

    memberships[~fg] = 0.0
    if np.any(labels[fg] < 0):
        raise RuntimeError("Calibrated CN11 left foreground pixels unclassified")
    if np.any(memberships[fg] < -1e-12):
        raise RuntimeError("Calibrated CN11 produced negative membership")
    if not np.allclose(memberships[fg].sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Calibrated CN11 memberships do not sum to one")
    return labels, memberships, probabilities, raw_winner, lab, chroma, hue


def _bbox_slices(mask: np.ndarray) -> Optional[Tuple[slice, slice]]:
    """Bounding box of the True region of a 2D boolean mask, as
    (row_slice, col_slice); None if the mask has no True pixel at all."""

    rows = np.any(mask, axis=1)
    if not rows.any():
        return None
    cols = np.any(mask, axis=0)
    y_hits = np.where(rows)[0]
    x_hits = np.where(cols)[0]
    return slice(int(y_hits[0]), int(y_hits[-1]) + 1), slice(int(x_hits[0]), int(x_hits[-1]) + 1)


def _classify_delhey_rgb_details(
    rgb: np.ndarray,
    foreground: Optional[np.ndarray] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Same contract as _classify_delhey_rgb_details_impl, run only on the
    bounding box of `foreground` instead of the whole canvas.

    Every operation _impl performs (CN11 lookup, Lab conversion, every
    hue/chroma/lightness override, every soft membership transfer) is
    strictly per-pixel - none of them look at a neighbouring pixel - so
    cropping to the bounding box before running them and embedding the
    result back into full-canvas arrays afterwards is exact, not an
    approximation: foreground pixels get bit-identical results, and every
    caller already discards everything outside `foreground` anyway. On a
    specimen whose mask only covers a fraction of the 2024x2024 canvas
    (the norm here - see README "Ce que cette normalisation fait au
    signal"), this is most of where classify_delhey_rgb's ~12s/image goes.
    """

    arr = _check_rgb(rgb)
    fg = _check_foreground(foreground, arr.shape[:2])

    bbox = _bbox_slices(fg)
    if bbox is None:  # no foreground pixel at all - degenerate, nothing to crop to
        return _classify_delhey_rgb_details_impl(arr, fg)

    rows, cols = bbox
    cropped = _classify_delhey_rgb_details_impl(arr[rows, cols], fg[rows, cols])
    cropped_labels, cropped_memberships, cropped_probabilities, cropped_raw_winner, cropped_lab, cropped_chroma, cropped_hue = cropped

    shape = arr.shape[:2]
    labels = np.full(shape, -1, dtype=cropped_labels.dtype)
    memberships = np.zeros((*shape, cropped_memberships.shape[-1]), dtype=cropped_memberships.dtype)
    probabilities = np.zeros((*shape, cropped_probabilities.shape[-1]), dtype=cropped_probabilities.dtype)
    raw_winner = np.full(shape, -1, dtype=cropped_raw_winner.dtype)
    lab = np.zeros((*shape, 3), dtype=cropped_lab.dtype)
    chroma = np.zeros(shape, dtype=cropped_chroma.dtype)
    hue = np.zeros(shape, dtype=cropped_hue.dtype)

    labels[rows, cols] = cropped_labels
    memberships[rows, cols] = cropped_memberships
    probabilities[rows, cols] = cropped_probabilities
    raw_winner[rows, cols] = cropped_raw_winner
    lab[rows, cols] = cropped_lab
    chroma[rows, cols] = cropped_chroma
    hue[rows, cols] = cropped_hue

    return labels, memberships, probabilities, raw_winner, lab, chroma, hue


def classify_delhey_rgb(
    rgb: np.ndarray, foreground: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return hard CN11 labels after documented local boundary calibrations.

    Returns an integer array; -1 denotes alpha-defined background and
    non-negative integers index :data:`DELHEY_CATEGORIES`.
    """

    arr = _check_rgb(rgb)
    fg = _check_foreground(foreground, arr.shape[:2])
    labels, _, _, _, _, _, _ = _classify_delhey_rgb_details(arr, fg)
    return labels


def ablate_delhey_colour(
    rgb: np.ndarray,
    category: str,
    foreground: Optional[np.ndarray] = None,
    strength: float = 1.0,
    mask_feather_sigma: float = 0.0,
    mask_mode: str = "hard",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Remove one chromatic category while preserving pixel L*.

    ``mask_mode='hard'`` is the primary intervention and uses the calibrated
    categorical label. ``mask_mode='soft'`` is a sensitivity analysis: a pixel
    with membership 0.8 retains 20% of its original a*/b* chroma at strength 1.

    A thin wrapper around ablate_delhey_colour_from_details(): classifies
    once, then applies. Call that function directly instead when you also
    need to know which categories are present (e.g. to pick one) - it lets
    you reuse the same classification for both instead of running
    _classify_delhey_rgb_details() twice (src/ablations/a6_colour_removal.py
    does exactly that).
    """

    arr = _check_rgb(rgb)
    fg = _check_foreground(foreground, arr.shape[:2])
    details = _classify_delhey_rgb_details(arr, fg)
    return ablate_delhey_colour_from_details(
        arr, category, fg, details, strength=strength, mask_feather_sigma=mask_feather_sigma, mask_mode=mask_mode
    )


def ablate_delhey_colour_from_details(
    rgb: np.ndarray,
    category: str,
    foreground: np.ndarray,
    details: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    strength: float = 1.0,
    mask_feather_sigma: float = 0.0,
    mask_mode: str = "hard",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Same as ablate_delhey_colour(), but takes an already-computed
    classification (the tuple _classify_delhey_rgb_details() returns)
    instead of recomputing it - the actual removal math, cheap on its own.
    """

    if category not in DELHEY_CHROMATIC:
        raise ValueError(
            "Colour ablation is defined only for the eight chromatic categories: "
            + ", ".join(DELHEY_CHROMATIC_CATEGORIES)
        )
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    if mask_feather_sigma < 0:
        raise ValueError("mask_feather_sigma must be non-negative")
    if mask_mode not in {"hard", "soft"}:
        raise ValueError("mask_mode must be 'hard' or 'soft'")

    labels, memberships, probabilities, raw_winner, lab, chroma, hue = details
    fg = foreground
    category_index = DELHEY_CATEGORIES.index(category)
    hard_mask = labels == category_index
    membership_mask = memberships[..., category_index]
    applied_mask = (
        hard_mask.astype(np.float64)
        if mask_mode == "hard"
        else membership_mask.copy()
    )
    if mask_feather_sigma > 0 and np.any(applied_mask > 0):
        applied_mask = ndimage.gaussian_filter(applied_mask, mask_feather_sigma)
        if mask_mode == "hard":
            maximum = float(applied_mask.max())
            if maximum > 0:
                applied_mask /= maximum
        applied_mask = np.clip(applied_mask, 0.0, 1.0)
        applied_mask *= fg

    out = lab.copy()
    out[..., 1] = lab[..., 1] * (1.0 - strength * applied_mask)
    out[..., 2] = lab[..., 2] * (1.0 - strength * applied_mask)

    hard_fraction = float(hard_mask.sum() / max(1, fg.sum()))
    mean_membership = float(membership_mask[fg].mean())
    raw_scores = probabilities[fg]
    raw_ordered = np.sort(raw_scores, axis=1)
    raw_confidence = raw_ordered[:, -1]
    raw_margin = raw_ordered[:, -1] - raw_ordered[:, -2]
    base_labels, _ = _base_cn11_labels_and_memberships(
        probabilities, raw_winner, lab[..., 0], hue, fg
    )
    override_fraction = float(np.mean(labels[fg] != base_labels[fg]))
    metadata = {
        "classifier": "CN11_locally_calibrated_to_Delhey_12_v6",
        "mask_mode": mask_mode,
        "category_fraction_foreground": hard_fraction,
        "mean_category_membership_foreground": mean_membership,
        "mean_applied_mask_foreground": float(applied_mask[fg].mean()),
        "fraction_applied_mask_gt_0_5": float(np.mean(applied_mask[fg] > 0.5)),
        "mean_chroma_in_category": (
            float(chroma[hard_mask].mean()) if np.any(hard_mask) else 0.0
        ),
        "ablation_strength": float(strength),
        "decision_rule": "CN11_baseline_plus_documented_local_CIELCh_calibrations_v6",
        "calibration_rules": dict(DELHEY_CN11_CALIBRATION),
        "fraction_hard_labels_changed_from_base_CN11_mapping": override_fraction,
        "mean_raw_cn11_confidence_foreground": float(raw_confidence.mean()),
        "mean_raw_cn11_margin_foreground": float(raw_margin.mean()),
        "fraction_raw_cn11_ambiguous_margin_lt_0_15": float(
            np.mean(raw_margin < 0.15)
        ),
        "mode": "chroma_to_neutral_preserve_L",
        "foreground_only": foreground is not None,
    }
    result = lab_to_rgb(out)
    # Lab round-tripping alone can change RGB values by small numerical amounts.
    # Restore alpha-defined background exactly so no colour ablation is ever
    # applied outside the bird, even indirectly through the conversion.
    result[~fg] = _check_rgb(rgb)[~fg]
    return result, (hard_mask if mask_mode == "hard" else applied_mask), metadata
