import numpy as np

# Joint order for MediaPipe 15 (MP15)
JOINT_LABELS = [
    "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Foot", "R_Foot",
]
ROW_MP15 = {name: i for i, name in enumerate(JOINT_LABELS)}

# Angle triplets (A, B, C) -> angle at B between BA and BC
_ANG_TRIPLETS = [
    (ROW_MP15["R_Shoulder"], ROW_MP15["L_Shoulder"], ROW_MP15["L_Elbow"]),
    (ROW_MP15["L_Shoulder"], ROW_MP15["L_Elbow"], ROW_MP15["L_Wrist"]),
    (ROW_MP15["L_Elbow"],    ROW_MP15["L_Shoulder"], ROW_MP15["L_Hip"]),
    (ROW_MP15["L_Shoulder"], ROW_MP15["L_Hip"], ROW_MP15["L_Knee"]),
    (ROW_MP15["L_Hip"],      ROW_MP15["L_Knee"], ROW_MP15["L_Ankle"]),
    (ROW_MP15["L_Knee"],     ROW_MP15["L_Ankle"], ROW_MP15["L_Foot"]),
    (ROW_MP15["L_Knee"],     ROW_MP15["L_Hip"], ROW_MP15["R_Hip"]),
    (ROW_MP15["L_Foot"],     ROW_MP15["L_Knee"], ROW_MP15["L_Shoulder"]),
    (ROW_MP15["L_Shoulder"], ROW_MP15["R_Shoulder"], ROW_MP15["R_Elbow"]),
    (ROW_MP15["R_Shoulder"], ROW_MP15["R_Elbow"], ROW_MP15["R_Wrist"]),
    (ROW_MP15["R_Elbow"],    ROW_MP15["R_Shoulder"], ROW_MP15["R_Hip"]),
    (ROW_MP15["R_Shoulder"], ROW_MP15["R_Hip"], ROW_MP15["R_Knee"]),
    (ROW_MP15["R_Hip"],      ROW_MP15["R_Knee"], ROW_MP15["R_Ankle"]),
    (ROW_MP15["R_Knee"],     ROW_MP15["R_Ankle"], ROW_MP15["R_Foot"]),
    (ROW_MP15["R_Knee"],     ROW_MP15["R_Hip"], ROW_MP15["L_Hip"]),
    (ROW_MP15["R_Foot"],     ROW_MP15["R_Knee"], ROW_MP15["R_Shoulder"]),
]
_ANG_TRIPLETS = np.array(_ANG_TRIPLETS, dtype=int)


def compute_angles_mp15(coords: np.ndarray) -> np.ndarray:
    eps=1e-6
    if coords.ndim != 3 or coords.shape[1] < len(JOINT_LABELS):
        raise ValueError("coords must have shape (T, 15, 3)")

    pA = coords[:, _ANG_TRIPLETS[:, 0]]
    pB = coords[:, _ANG_TRIPLETS[:, 1]]
    pC = coords[:, _ANG_TRIPLETS[:, 2]]

    v1 = pA - pB
    v2 = pC - pB
    dot = np.einsum("tij,tij->ti", v1, v2)
    norm = np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2)
    cos = dot / (norm + 1e-8)

    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross, axis=2)
    sin = cross_norm / (norm + 1e-8)
    
    cos = np.clip(cos, -1.0, 1.0)
    sin = np.clip(sin, -1.0, 1.0)

    return cos.astype(np.float32), sin.astype(np.float32)
