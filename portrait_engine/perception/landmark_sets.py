"""Ordered MediaPipe FaceMesh landmark index rings.

MediaPipe 1.x removed the ``mediapipe.solutions`` package, so ``FACEMESH_*`` can no longer
be imported at runtime. These tuples were derived once from
``mediapipe/python/solutions/face_mesh_connections.py`` (Apache-2.0, Google LLC) by walking
each connection set into an ordered closed ring, and are frozen here. See
``docs/THIRD_PARTY.md``.

Every ring is ordered so that ``cv2.fillPoly`` produces the expected filled region without
self-intersection. Indices refer to the 478-point Face Landmarker output (468 mesh points
plus 10 iris points).

Naming follows MediaPipe: **left** and **right** are the subject's own left and right, which
appear mirrored in the image.
"""

from __future__ import annotations

#: Total landmarks produced by the Face Landmarker task with iris refinement.
LANDMARK_COUNT: int = 478

#: Closed face contour. Follows the jaw and roughly the hairline.
FACE_OVAL: tuple[int, ...] = (
    10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338,
)
#: Vermilion border (outer lip edge).
LIPS_OUTER: tuple[int, ...] = (
    0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
    267,
)
#: Inner lip edge. Its interior is the mouth opening.
LIPS_INNER: tuple[int, ...] = (
    13, 82, 81, 80, 191, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311,
    312,
)
#: Left eye aperture (eyelid margins).
LEFT_EYE: tuple[int, ...] = (
    249, 263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390,
)
#: Right eye aperture (eyelid margins).
RIGHT_EYE: tuple[int, ...] = (
    7, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163,
)
#: Left iris ring, from the refined-landmark model.
LEFT_IRIS: tuple[int, ...] = (
    474, 475, 476, 477,
)
#: Right iris ring, from the refined-landmark model.
RIGHT_IRIS: tuple[int, ...] = (
    469, 470, 471, 472,
)

# ---------------------------------------------------------------------------
# Sets below are assembled from the same connection data, but they are paths or
# multi-component trees rather than single cycles, so the closing order is chosen here.
# ---------------------------------------------------------------------------

#: Left eyebrow as a closed polygon: upper margin left-to-right, then lower margin back.
LEFT_EYEBROW: tuple[int, ...] = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)

#: Right eyebrow as a closed polygon, same convention.
RIGHT_EYEBROW: tuple[int, ...] = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)

#: Lower eyelid margin of the left eye, ordered outer corner -> inner corner.
LEFT_LOWER_LID: tuple[int, ...] = (263, 249, 390, 373, 374, 380, 381, 382, 362)

#: Lower eyelid margin of the right eye, ordered outer corner -> inner corner.
RIGHT_LOWER_LID: tuple[int, ...] = (33, 7, 163, 144, 145, 153, 154, 155, 133)

#: Eye corners, used for scale and for the inclination angle of derived regions.
LEFT_EYE_OUTER: int = 263
LEFT_EYE_INNER: int = 362
RIGHT_EYE_OUTER: int = 33
RIGHT_EYE_INNER: int = 133

#: Mouth corners.
MOUTH_LEFT_CORNER: int = 291
MOUTH_RIGHT_CORNER: int = 61

#: Nose region as a closed polygon: bridge midline down the subject's right alar crease,
#: across the columella, and back up the left alar crease.
NOSE: tuple[int, ...] = (
    168, 6, 197, 195, 5, 45, 220, 115, 48, 64, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275,
)

#: Alar base landmarks. The nostril openings sit just inside these, and are refined by
#: darkness rather than by geometry alone.
NOSTRIL_ANCHORS: tuple[int, ...] = (98, 327, 97, 326, 2)

#: Nose tip and bridge, used to detect specular highlights on the nose.
NOSE_TIP: int = 1
NOSE_BRIDGE: int = 6

#: Chin and glabella, used for the vertical extent of the face.
CHIN: int = 152
FOREHEAD_TOP: int = 10
GLABELLA: int = 9

#: Patch centers that are skin with very high confidence on any face: both cheeks, the
#: forehead and the nose bridge. These seed the per-image skin color model, so they must
#: never land on eyes, brows, lips or nostrils.
SKIN_SEED_POINTS: tuple[int, ...] = (
    205, 425, 50, 280, 234, 454, 151, 108, 337, 6, 118, 347,
)

#: Cheek centers, used as the luminance reference for under-eye correction.
LEFT_CHEEK: tuple[int, ...] = (425, 280, 371, 266)
RIGHT_CHEEK: tuple[int, ...] = (205, 50, 142, 36)

#: Forehead, nose and cheek groups for the lighting tool's highlight analysis.
FOREHEAD_REGION: tuple[int, ...] = (10, 338, 297, 332, 109, 67, 103, 151, 108, 337, 9)

ALL_RINGS: dict[str, tuple[int, ...]] = {
    "face_oval": FACE_OVAL,
    "lips_outer": LIPS_OUTER,
    "lips_inner": LIPS_INNER,
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "left_iris": LEFT_IRIS,
    "right_iris": RIGHT_IRIS,
    "left_eyebrow": LEFT_EYEBROW,
    "right_eyebrow": RIGHT_EYEBROW,
    "nose": NOSE,
}
