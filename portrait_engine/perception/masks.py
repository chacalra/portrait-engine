"""Mask construction: from 478 landmarks plus the image itself to a full ``MaskSet``.

Three sources of information are combined:

1. **Geometry.** Closed landmark rings are filled into polygons (face oval, eyes, irises,
   eyebrows, lips, mouth interior, nose). This is reliable but knows nothing about what is
   actually skin.
2. **Per-image skin color model.** A Gaussian in CIELAB seeded from patches that are skin
   with high confidence (cheeks, forehead, nose bridge) gives ``P_skin``. This is what makes
   the engine work across skin tones and lighting without recalibration, and it is what pulls
   the skin mask off the hairline and off a beard. Simpler than the GMM used by FabSoften and
   adapted to the specific photo. See ``docs/RESEARCH.md`` section 4.2.
3. **Local texture.** Hair, beard and eyelashes are dark and have much higher local
   high-frequency energy than skin. That gives a ``hair_like`` mask without a parsing model.

Derived regions that have no landmark ring of their own (the infraorbital region, the teeth)
are built here, following the contour-interpolation idea from the OpenCV G-API sample.

Every mask leaves this module as ``float32`` in ``[0, 1]`` and already feathered.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from portrait_engine.core import ops
from portrait_engine.core.colorspace import rgb_to_lab
from portrait_engine.core.ops import box_filter
from portrait_engine.core.types import FaceGeometry, MaskSet
from portrait_engine.perception import landmark_sets as ls
from portrait_engine.perception.landmarks import DetectedFace

#: Minimum variance floor for the L* channel of the skin model. Shading across a face spans
#: far more luminance than skin-tone variation does, so without this floor a lit forehead and
#: a shadowed jaw would be judged "not skin".
SKIN_L_VARIANCE_FLOOR = 14.0**2

#: Mahalanobis distance at which skin probability reaches ~0.5.
SKIN_SIGMA_SCALE = 2.4


@dataclass(frozen=True)
class MaskBuildConfig:
    """Tunables for mask construction. Fractions are of the interocular distance."""

    eye_dilate: float = 0.055
    eyebrow_dilate: float = 0.075
    lips_dilate: float = 0.045
    nostril_radius: float = 0.075
    under_eye_depth: float = 0.42  # fraction of eye width
    under_eye_reference_depth: float = 1.05  # fraction of eye width, for the cheek reference
    under_eye_lid_offset: float = 0.06  # push the top edge off the lashes
    under_eye_medial_extension: float = 0.22  # past the inner canthus, along the tear trough
    under_eye_medial_drop: float = 0.60  # how steeply that extension angles toward the chin
    under_eye_feather: float = 0.15  # fraction of the region's own depth, not of interocular
    under_eye_chroma_floor: float = 3.5  # a*/b* standard deviation floor, see the gate below
    under_eye_chroma_threshold: float = 0.15  # reject below this, then pass at full strength
    feather_radius: float = 0.075
    skin_erode: float = 0.02
    hair_texture_radius: float = 0.045
    hair_rim_width: float = 0.16


def geometry_from_landmarks(landmarks: np.ndarray) -> FaceGeometry:
    """Scale and pose descriptors. Purely read-only, no tool may alter these."""
    left_outer = landmarks[ls.LEFT_EYE_OUTER]
    left_inner = landmarks[ls.LEFT_EYE_INNER]
    right_outer = landmarks[ls.RIGHT_EYE_OUTER]
    right_inner = landmarks[ls.RIGHT_EYE_INNER]
    left_center = (left_outer + left_inner) / 2.0
    right_center = (right_outer + right_inner) / 2.0

    interocular = float(np.linalg.norm(left_center - right_center))
    oval = landmarks[list(ls.FACE_OVAL)]
    width = float(oval[:, 0].max() - oval[:, 0].min())
    height = float(oval[:, 1].max() - oval[:, 1].min())
    delta = left_center - right_center
    roll = float(np.degrees(np.arctan2(delta[1], delta[0])))

    return FaceGeometry(
        interocular_px=max(interocular, 1.0),
        face_width_px=max(width, 1.0),
        face_height_px=max(height, 1.0),
        roll_degrees=roll,
        eye_left_center=(float(left_center[0]), float(left_center[1])),
        eye_right_center=(float(right_center[0]), float(right_center[1])),
    )


def _fill_ring(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [np.rint(points).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    return (canvas.astype(np.float32) / 255.0)


def _ring(landmarks: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return landmarks[list(indices)]


def _disc(shape: tuple[int, int], center: np.ndarray, radius: int) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.circle(canvas, (int(round(center[0])), int(round(center[1]))), max(1, radius), 255, -1, cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0


def skin_color_probability(
    rgb: np.ndarray,
    seed_mask: np.ndarray,
    *,
    l_variance_floor: float = SKIN_L_VARIANCE_FLOOR,
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-image skin probability from a Gaussian fitted to confident skin samples.

    Robustness: the covariance is estimated twice, the second time after discarding samples
    beyond 2.5 sigma, so a stray dark pixel in a seed patch does not widen the model.

    Returns:
        ``(probability in [0,1], stats)``.
    """
    lab = rgb_to_lab(rgb)
    samples = lab[seed_mask > 0.5]
    if samples.shape[0] < 64:
        # Not enough confident skin. Fall back to "everything inside the seed is skin".
        return np.clip(seed_mask, 0.0, 1.0), {"samples": float(samples.shape[0]), "fallback": 1.0}

    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False) + np.eye(3) * 1e-3
    inv = np.linalg.inv(cov)
    d2 = np.einsum("ij,jk,ik->i", samples - mean, inv, samples - mean)
    keep = d2 < 2.5**2
    if keep.sum() > 64:
        samples = samples[keep]
        mean = samples.mean(axis=0)
        cov = np.cov(samples, rowvar=False) + np.eye(3) * 1e-3

    cov[0, 0] = max(cov[0, 0], l_variance_floor)
    inv = np.linalg.inv(cov)

    flat = lab.reshape(-1, 3) - mean
    d2_all = np.einsum("ij,jk,ik->i", flat, inv, flat).reshape(lab.shape[:2])
    prob = np.exp(-0.5 * d2_all / (SKIN_SIGMA_SCALE**2 / 2.0)).astype(np.float32)

    stats = {
        "samples": float(samples.shape[0]),
        "mean_L": float(mean[0]),
        "mean_a": float(mean[1]),
        "mean_b": float(mean[2]),
        "std_L": float(np.sqrt(cov[0, 0])),
        "fallback": 0.0,
    }
    return np.clip(prob, 0.0, 1.0), stats


def infraorbital_color_gate(
    lab: np.ndarray,
    reference_band: np.ndarray,
    config: MaskBuildConfig,
) -> np.ndarray:
    """Reject non-skin content from the infraorbital region without penalising the dark circle.

    The general ``skin_color_probability`` must not be used here. It is seeded from cheek,
    forehead and nose-bridge patches, so a region that is darker and often bluer scores as
    "less skin" precisely in proportion to how bad the dark circle is. Measured on a real
    portrait, that gate cut the region from 1735 px to 760 px: the worse the defect, the
    weaker the mask meant to treat it.

    Two changes make this gate safe to use on the defect itself:

    1. **Chroma only.** ``L*`` is dropped, because darkness *is* the thing being corrected.
       Containment against hair, brows and lashes already comes from the anatomical polygon,
       the face oval and the ``protected`` union, so this gate is defence in depth against
       eyeglass frames and stray strands, not the primary bound.
    2. **Hard rejection, not a soft weight.** Below the threshold the pixel is excluded;
       above it, it passes at full strength. A soft probability silently scaled the whole
       correction down (peak alpha 0.70 instead of 1.00) while rejecting nothing.

    The covariance is fitted to this eye's own cheek reference band, so it adapts to the
    subject and to the white balance, and gets a standard-deviation floor for the same reason
    ``SKIN_L_VARIANCE_FLOOR`` exists: a smooth cheek has a very tight chroma distribution, and
    without the floor the Mahalanobis distance explodes over perfectly normal skin. Measured
    across six portraits, the floor raised the region kept on the deepest dark circle from
    70.8 percent to 99.6 percent while rejecting exactly as much hair-like content.
    """
    samples = lab[reference_band > 0.5][:, 1:]
    if samples.shape[0] < 64:
        return np.ones(lab.shape[:2], dtype=np.float32)

    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False) + np.eye(2) * 1e-2
    floor = config.under_eye_chroma_floor**2
    cov[0, 0] = max(cov[0, 0], floor)
    cov[1, 1] = max(cov[1, 1], floor)

    flat = lab[..., 1:].reshape(-1, 2) - mean
    d2 = np.einsum("ij,jk,ik->i", flat, np.linalg.inv(cov), flat).reshape(lab.shape[:2])
    prob = np.exp(-0.5 * d2 / (SKIN_SIGMA_SCALE**2 / 2.0))
    return (prob > config.under_eye_chroma_threshold).astype(np.float32)


def face_axis_coordinate(
    shape: tuple[int, int], landmarks: np.ndarray
) -> tuple[np.ndarray, float]:
    """Normalized coordinate along the face's own vertical axis (glabella 0, chin 1).

    Computed by projection onto the glabella-to-chin vector rather than by image rows, so it
    stays correct when the head is rolled. Returns ``(coordinate map, nose base coordinate)``.
    """
    top = landmarks[ls.GLABELLA].astype(np.float32)
    bottom = landmarks[ls.CHIN].astype(np.float32)
    axis = bottom - top
    length = float(np.linalg.norm(axis))
    if length < 1e-3:
        return np.zeros(shape, dtype=np.float32), 0.5
    unit = axis / length

    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    coordinate = ((xs - top[0]) * unit[0] + (ys - top[1]) * unit[1]) / length
    nose_base = float(np.dot(landmarks[2].astype(np.float32) - top, unit) / length)
    return coordinate.astype(np.float32), nose_base


def hair_like_mask(
    rgb: np.ndarray,
    lab: np.ndarray,
    face_oval: np.ndarray,
    skin_reference: np.ndarray,
    landmarks: np.ndarray,
    geometry: FaceGeometry,
    config: MaskBuildConfig,
) -> np.ndarray:
    """Detect hair, beard, stubble and eyelashes without a semantic segmentation model.

    Two ingredients, and both are needed.

    **1. An anatomical prior.** Hair can only be in two places inside the face oval: the beard
    zone (everything below the base of the nose, which covers moustache, chin and jaw) and the
    rim of the oval (hairline, temples, sideburns). Nothing in the middle of a cheek, the
    forehead or the nose is ever hair. Without this prior, the texture test below flags the
    shadow side of a side-lit face: on one test portrait an earlier version cut a
    dark-skinned subject's skin mask in half that way.

    **2. Relative local contrast**, ``|L - smooth(L)| / (smooth(L) + k)``, not absolute. This
    was measured, not assumed. A very dark beard is crushed toward black, so its *absolute*
    luminance variation is smaller than a lit cheek's: on one portrait the beard measured an
    absolute texture ratio of 0.75 against skin, which the obvious absolute test misses
    entirely. In relative terms the same beard measures 3.0x skin, because hairs and the gaps
    between them differ by a large fraction of a very small mean. Measured probe values were
    2.98 and 4.30 for beard against 0.53 to 1.60 for skin.

    Three detection terms are combined with a max, then multiplied by the prior:

    * textured **and** darker than skin: normal hair and mid-tone beards;
    * strongly textured at skin luminance: grey or blond beards, light hair;
    * crushed to near-black: deep beard shadow the texture term underestimates.

    Known limitation: a clean-shaven jaw in deep shadow is partly flagged, which costs a
    little skin-smoothing coverage in a region that has nothing to smooth. This is the weakest
    part of the perception layer and the first thing the optional BiSeNet backend improves.
    """
    lightness = lab[..., 0]
    radius = geometry.scaled(config.hair_texture_radius, 2)
    base = ops.guided_filter(lightness, lightness, radius, 4.0, subsample=2)
    local_mean = box_filter(lightness, radius)
    relative_energy = box_filter(np.abs(lightness - base), radius) / (local_mean + 6.0)

    skin_energy = ops.masked_percentile(relative_energy, skin_reference, 60.0)
    skin_lightness = ops.masked_percentile(lightness, skin_reference, 50.0)
    if not np.isfinite(skin_energy) or skin_energy <= 1e-6:
        skin_energy = float(max(relative_energy.mean(), 1e-6))
    if not np.isfinite(skin_lightness) or skin_lightness <= 1.0:
        skin_lightness = float(max(lightness.mean(), 1.0))

    ratio = relative_energy / skin_energy
    deficit = skin_lightness - lightness

    textured_dark = ops.smoothstep(ratio, 1.8, 3.2) * ops.smoothstep(deficit, 2.0, 13.0)
    textured_light = ops.smoothstep(ratio, 3.2, 6.0)
    crushed = ops.smoothstep(deficit, 0.80 * skin_lightness, 0.92 * skin_lightness)
    detected = np.maximum.reduce([textured_dark, textured_light, crushed])

    axis, nose_base = face_axis_coordinate(rgb.shape[:2], landmarks)
    beard_zone = ops.smoothstep(axis, nose_base - 0.04, nose_base + 0.10)
    rim_width = geometry.scaled(config.hair_rim_width, 2)
    oval_rim = np.clip(face_oval - ops.erode_mask(face_oval, rim_width), 0.0, 1.0)
    prior = np.clip(np.maximum(beard_zone, oval_rim), 0.0, 1.0)

    mask = detected * prior * face_oval
    mask = ops.dilate_mask(mask, geometry.scaled(0.02, 1))
    return ops.blur_mask(mask, geometry.scaled(0.02, 1))


def under_eye_polygon(
    landmarks: np.ndarray,
    lid_indices: tuple[int, ...],
    chin: np.ndarray,
    geometry: FaceGeometry,
    config: MaskBuildConfig,
) -> np.ndarray:
    """Build the infraorbital contour by offsetting the lower eyelid toward the cheek.

    The 478-point mesh has no infraorbital ring, so this is the one region that has to be
    interpolated, the same way the OpenCV G-API sample interpolates eye and forehead contours
    from a handful of points.

    The offset depth follows a sine profile along the lid: shallow at both corners, deepest
    under the middle of the eye, which is where the shadow actually sits. Direction is the
    lid normal, oriented toward the chin so the polygon works on a rolled or upside-down face.
    """
    lid = landmarks[list(lid_indices)].astype(np.float32)
    outer, inner = lid[0], lid[-1]
    axis = inner - outer
    eye_width = float(np.linalg.norm(axis))
    if eye_width < 1e-3:
        return lid

    unit = axis / eye_width
    normal = np.array([-unit[1], unit[0]], dtype=np.float32)
    lid_center = lid.mean(axis=0)
    if float(np.dot(chin - lid_center, normal)) < 0:
        normal = -normal

    depth = config.under_eye_depth * eye_width
    t = np.linspace(0.0, 1.0, lid.shape[0], dtype=np.float32)
    profile = 0.45 + 0.55 * np.sin(np.pi * np.power(t, 0.85))

    # Extend past the inner canthus along the tear trough. The lower-lid ring ends at the
    # canthus, but the nasojugal fold continues down and toward the nose, and that is often the
    # darkest part of the whole area: measured on one portrait, 550 px at a 16.1 L* deficit
    # sitting entirely outside the region, against 18.7 L* inside it. Without this the tool
    # brightens the flat infraorbital area and leaves the trough, which reads as an unfinished
    # retouch.
    #
    # Kept short and tapered on purpose. Further medially and steeper lies the nasal sidewall
    # shadow, which is facial *form*, not a blemish; brightening it flattens the nose.
    path, prof = lid, profile
    extension = config.under_eye_medial_extension * eye_width
    if extension > 0.0:
        direction = unit + normal * config.under_eye_medial_drop
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        steps = np.linspace(0.0, 1.0, max(2, lid.shape[0] // 3) + 1, dtype=np.float32)[1:]
        path = np.concatenate([lid, inner + direction * (extension * steps)[:, None]])
        prof = np.concatenate([profile, profile[-1] * (1.0 - 0.55 * steps)])

    top = path + normal * (config.under_eye_lid_offset * eye_width)
    bottom = path + normal * (depth * prof)[:, None]
    return np.concatenate([top, bottom[::-1]], axis=0)


def teeth_mask(
    lab: np.ndarray,
    mouth_interior: np.ndarray,
    lips: np.ndarray,
    geometry: FaceGeometry,
) -> tuple[np.ndarray, dict[str, float]]:
    """Segment teeth inside the mouth opening using lips as the red reference.

    Teeth are the bright, low-chroma content of the mouth opening. Gums, tongue and the inner
    lip are red, and the gap between the lips is dark. Thresholds are taken relative to the
    measured lip color rather than being absolute, so this works on any skin tone and under
    any white balance.

    Deliberately conservative: if the result covers too little of the mouth opening, it is
    reported as not found so that ``whiten_teeth`` becomes a no-op instead of bleaching a
    tongue.
    """
    stats: dict[str, float] = {}
    area = float(mouth_interior.sum())
    stats["mouth_area_px"] = area
    min_area = max(24.0, (geometry.interocular_px * 0.06) ** 2)
    if area < min_area:
        stats["found"] = 0.0
        stats["reason_small_mouth"] = 1.0
        return np.zeros_like(mouth_interior), stats

    lightness, a_star = lab[..., 0], lab[..., 1]

    l_high = ops.masked_percentile(lightness, mouth_interior, 92.0)
    a_lip = ops.masked_percentile(a_star, lips, 55.0)
    if not np.isfinite(l_high) or not np.isfinite(a_lip):
        stats["found"] = 0.0
        return np.zeros_like(mouth_interior), stats
    a_lip = max(a_lip, 4.0)

    # Two tests only: bright, and not red.
    #
    # A "low chroma" term was tried first and removed. Chroma in general is not a
    # teeth-versus-not-teeth signal: yellow teeth have plenty of it, and the term rejected
    # exactly the discolored areas the tool exists to correct. Measured on a subject with
    # yellow teeth, it reduced the mask to the whitish specular highlights on each tooth. What
    # actually separates teeth from gums, tongue and inner lip is **redness**, which is the
    # ``a*`` axis, and that is what the second term measures.
    bright = ops.smoothstep(lightness, 0.38 * l_high, 0.66 * l_high)
    not_red = 1.0 - ops.smoothstep(a_star, 0.42 * a_lip, 0.88 * a_lip)

    score = bright * not_red * mouth_interior
    score = ops.erode_mask(score, geometry.scaled(0.012, 1))
    score = ops.blur_mask(score, geometry.scaled(0.012, 1))
    score = np.clip(score * mouth_interior, 0.0, 1.0)

    stats["teeth_area_px"] = float(score.sum())
    stats["mouth_coverage"] = float(score.sum() / max(area, 1.0))
    stats["l_high"] = float(l_high)
    stats["a_lip"] = float(a_lip)
    stats["width_ratio"] = _width_ratio(score, mouth_interior)
    stats["found"] = 1.0 if score.sum() > 1.0 else 0.0
    return score.astype(np.float32), stats


def _width_ratio(mask: np.ndarray, reference: np.ndarray) -> float:
    """Horizontal extent of ``mask`` as a share of ``reference``'s extent.

    Scale invariant, unlike an area share, which is why the teeth confidence test uses it: a
    real set of teeth spans the mouth opening, a stray bright blob does not.
    """
    columns = np.where(mask.max(axis=0) > 0.35)[0]
    reference_columns = np.where(reference.max(axis=0) > 0.5)[0]
    if columns.size == 0 or reference_columns.size < 2:
        return 0.0
    span = float(columns.max() - columns.min() + 1)
    reference_span = float(reference_columns.max() - reference_columns.min() + 1)
    return span / max(reference_span, 1.0)


def build_masks(
    rgb: np.ndarray,
    face: DetectedFace,
    geometry: FaceGeometry,
    config: MaskBuildConfig | None = None,
) -> tuple[MaskSet, np.ndarray, dict[str, float]]:
    """Build the complete mask set for one face.

    Returns:
        ``(masks, skin_probability, stats)``.
    """
    config = config or MaskBuildConfig()
    shape = (rgb.shape[0], rgb.shape[1])
    landmarks = face.landmarks
    lab = rgb_to_lab(rgb)
    stats: dict[str, float] = {}

    feather = geometry.scaled(config.feather_radius, 2)

    # --- geometric regions -------------------------------------------------------------
    face_oval = _fill_ring(shape, _ring(landmarks, ls.FACE_OVAL))
    left_eye = _fill_ring(shape, _ring(landmarks, ls.LEFT_EYE))
    right_eye = _fill_ring(shape, _ring(landmarks, ls.RIGHT_EYE))
    left_brow = _fill_ring(shape, _ring(landmarks, ls.LEFT_EYEBROW))
    right_brow = _fill_ring(shape, _ring(landmarks, ls.RIGHT_EYEBROW))
    lips_outer = _fill_ring(shape, _ring(landmarks, ls.LIPS_OUTER))
    lips_inner = _fill_ring(shape, _ring(landmarks, ls.LIPS_INNER))
    nose = _fill_ring(shape, _ring(landmarks, ls.NOSE))

    eyes_dilated = ops.dilate_mask(
        np.maximum(left_eye, right_eye), geometry.scaled(config.eye_dilate, 2)
    )
    brows_dilated = ops.dilate_mask(
        np.maximum(left_brow, right_brow), geometry.scaled(config.eyebrow_dilate, 2)
    )
    lips = np.clip(lips_outer - lips_inner, 0.0, 1.0)
    lips_dilated = ops.dilate_mask(lips_outer, geometry.scaled(config.lips_dilate, 2))

    # Nostril openings are the dark content of the lower part of the nose region. Deriving
    # them from luminance instead of from anchor landmarks is more accurate, because the
    # alar-crease landmarks sit beside the openings rather than on them.
    nose_lower = nose.copy()
    nose_points = _ring(landmarks, ls.NOSE)
    nose_top, nose_bottom = float(nose_points[:, 1].min()), float(nose_points[:, 1].max())
    split = nose_top + 0.55 * (nose_bottom - nose_top)
    rows = np.arange(shape[0], dtype=np.float32)[:, None]
    nose_lower = nose_lower * ops.smoothstep(
        np.broadcast_to(rows, shape), split, split + 0.12 * geometry.interocular_px
    )
    nose_l = ops.masked_percentile(lab[..., 0], nose, 60.0)
    if np.isfinite(nose_l):
        darkness = ops.smoothstep(nose_l - lab[..., 0], 6.0, 22.0)
    else:
        darkness = np.zeros(shape, dtype=np.float32)
    nostrils = nose_lower * darkness
    nostrils = ops.dilate_mask(nostrils, geometry.scaled(config.nostril_radius * 0.35, 1))
    nostrils = ops.blur_mask(nostrils, geometry.scaled(0.02, 1))

    # --- skin color model --------------------------------------------------------------
    seed_r = geometry.scaled(0.075, 3)
    seed = np.zeros(shape, dtype=np.float32)
    for index in ls.SKIN_SEED_POINTS:
        seed = np.maximum(seed, _disc(shape, landmarks[index], seed_r))
    seed = seed * ops.erode_mask(face_oval, geometry.scaled(config.skin_erode, 1))
    seed = np.clip(seed - eyes_dilated - brows_dilated - lips_dilated - nostrils, 0.0, 1.0)

    skin_prob, prob_stats = skin_color_probability(rgb, seed)
    stats.update({f"skin_{k}": v for k, v in prob_stats.items()})

    hair_like = hair_like_mask(rgb, lab, face_oval, seed, landmarks, geometry, config)

    # --- protected union ----------------------------------------------------------------
    protected = np.clip(
        np.maximum.reduce([eyes_dilated, brows_dilated, lips_dilated, nostrils, hair_like]),
        0.0,
        1.0,
    )
    protected = ops.blur_mask(protected, geometry.scaled(0.015, 1))

    # --- skin ---------------------------------------------------------------------------
    oval_inner = ops.erode_mask(face_oval, geometry.scaled(config.skin_erode, 1))
    skin_raw = oval_inner * skin_prob
    skin = np.clip(skin_raw * (1.0 - protected), 0.0, 1.0)
    skin = ops.feather_mask(skin, rgb, feather, eps=1e-3, subsample=2)
    # Re-apply both bounds after feathering. Guided feathering is a low-pass, so it pushes the
    # mask outward wherever the image has no edge to snap to: without this the skin mask reached
    # alpha 0.46 outside the face oval (measured, and caught by a test), which would let the
    # smoothing tool work on the neck and on the background near the jaw.
    containment = ops.blur_mask(
        ops.dilate_mask(face_oval, geometry.scaled(0.03, 1)), geometry.scaled(0.02, 1)
    )
    skin = np.clip(skin * containment * (1.0 - protected), 0.0, 1.0)
    stats["skin_area_px"] = float(skin.sum())

    # --- derived: infraorbital ----------------------------------------------------------
    chin = landmarks[ls.CHIN]
    under_eyes: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    for side, lid in (("left", ls.LEFT_LOWER_LID), ("right", ls.RIGHT_LOWER_LID)):
        polygon = under_eye_polygon(landmarks, lid, chin, geometry, config)

        # The cheek band just below the infraorbital region, used as the "healthy skin"
        # luminance and chroma reference. Built with the same interpolation but deeper, then
        # the under-eye region itself is removed, leaving a ring of neighbouring skin. Built
        # before the region because the region's color gate is fitted to it.
        deep = under_eye_polygon(
            landmarks, lid, chin, geometry,
            replace(config, under_eye_depth=config.under_eye_reference_depth),
        )
        band = np.clip(_fill_ring(shape, deep) - _fill_ring(shape, polygon), 0.0, 1.0)
        band = band * skin_prob * face_oval
        band = np.clip(band * (1.0 - protected), 0.0, 1.0)
        references[side] = ops.blur_mask(band, geometry.scaled(0.02, 1))
        stats[f"under_eye_{side}_reference_px"] = float(references[side].sum())

        region = _fill_ring(shape, polygon)
        region = np.clip(region - eyes_dilated, 0.0, 1.0)
        region = region * infraorbital_color_gate(lab, references[side], config) * face_oval
        region = np.clip(region * (1.0 - protected), 0.0, 1.0)
        # A plain blur, not `feather_mask`. Guided feathering makes alpha follow image edges,
        # so the alpha itself carries skin texture; compositing then imprints an inverted copy
        # of that texture, scaled by the lift. Measured on a texture-free +9 L* lift: guided
        # feathering scored 0.902 high-frequency retention, a plain blur 1.005. The
        # infraorbital boundary is anatomical and genuinely soft, so it needs no edge
        # awareness. The radius is a fraction of the region's OWN depth: at the previous
        # 0.09 * interocular the radius was about half the region height and collapsed the
        # peak alpha to 0.40, capping every correction at 40 percent of its nominal value.
        lid_points = landmarks[list(lid)].astype(np.float32)
        eye_width = float(np.linalg.norm(lid_points[-1] - lid_points[0]))
        depth_px = config.under_eye_depth * eye_width
        region = ops.blur_mask(region, max(3, int(round(depth_px * config.under_eye_feather))))
        # Feathering pushes the region back up into the eye, so the eye bound is re-applied.
        region = np.clip(region * (1.0 - eyes_dilated) * (1.0 - protected), 0.0, 1.0)
        under_eyes[side] = np.clip(region, 0.0, 1.0)
        stats[f"under_eye_{side}_area_px"] = float(region.sum())
        stats[f"under_eye_{side}_peak_alpha"] = float(region.max())

    # --- derived: teeth -----------------------------------------------------------------
    mouth_interior = ops.erode_mask(lips_inner, geometry.scaled(0.015, 1))
    teeth, teeth_stats = teeth_mask(lab, mouth_interior, lips, geometry)
    stats.update({f"teeth_{k}": v for k, v in teeth_stats.items()})

    # --- photometric region for the lighting tool ---------------------------------------
    # Wider than `skin`: it includes the nose and the chin so a low-frequency gain field does
    # not produce a visible seam, but still excludes eyes and lips.
    photometric = np.clip(
        ops.blur_mask(face_oval, geometry.scaled(0.05, 2)) * (1.0 - eyes_dilated) * (1.0 - lips_inner),
        0.0,
        1.0,
    )

    # The mask set is a published contract and the debug inspection surface, not merely what the
    # tools happen to read, so per-side masks stay even where only their union is consumed.
    #
    # Five that were neither in the contract nor read by anything were removed: `skin_wide`,
    # `eyes_dilated`, `left_iris`, `right_iris` and `lips_outer`. At native resolution each was
    # 46 MB of float32 held for the lifetime of every `FaceAnalysis`, so that is 231 MB back.
    # They are still built as local intermediates where needed and freed with this function.
    masks = MaskSet(
        {
            "face_oval": face_oval,
            "skin": skin,
            "protected": protected,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "eyes": np.maximum(left_eye, right_eye),
            "left_eyebrow": left_brow,
            "right_eyebrow": right_brow,
            "eyebrows": np.maximum(left_brow, right_brow),
            "lips": lips,
            "mouth_interior": mouth_interior,
            "teeth": teeth,
            "nose": nose,
            "nostrils": nostrils,
            "hair_like": hair_like,
            "left_under_eye": under_eyes["left"],
            "left_under_eye_reference": references["left"],
            "right_under_eye_reference": references["right"],
            "right_under_eye": under_eyes["right"],
            "under_eye": np.maximum(under_eyes["left"], under_eyes["right"]),
            "face_photometric": photometric,
        },
        shape,
    )
    return masks, skin_prob, stats
