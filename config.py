from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    RAW_DATA_DIR: Path = Path("data/raw")
    PROCESSED_DATA_DIR: Path = Path("data/processed")
    VELOCITY_THRESHOLD_REF_MODE: float = 0.01
    STILL_WINDOW: int = 5
    WINDOW_SIZE: int = 50
    STRIDE_LENGTH: int = 1
    DEFAULT_SCALE: float = 1.0
    SAVGOL_WINDOW: int = 9
    SAVGOL_POLY: int = 3
    NUM_WORKERS: int = 32
    ANGLE_OUTPUT_DIR: Path = Path("data/angles")

    LOG_LEVEL: str = "INFO"
    OUTPUT_ROOT: Path = Path("data/processed")
    RAW_ROOT: Path = Path("data/raw")
    COMPRESSION: str = "lzf"
    SPLIT_RATIOS: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    SYNTHETIC_DEPTH_RHOS: Tuple[float, float] = (0.75, 0.5)

    TRIPLET_WINDOW_SIZE: int = 50
    TRIPLET_STRIDE: int = 2
    TRIPLET_BATCH_SIZE: int = 16
    TRIPLET_LEARNING_RATE: float = 1e-4
    TRIPLET_EPOCHS: int = 15

    REF_WINDOW_SIZE: int = 50
    REF_STRIDE: int = 2
    REF_ENERGY_THRESHOLD_DEG: float = 5.0

    THR_SIM: float = 0.72
    THR_MARGIN: float = 0.10
    THR_JOINT: float = 0.20
    THR_ROM: float = 0.90
    THR_TEMPO: float = 0.30
    THR_DROP: float = 0.62
    THR_K_CONFIRM: int = 2
    THR_N_DROP: int = 3


class Landmark(IntEnum):
    

    HEAD = 0
    L_EYE_INNER = 1
    L_EYE = 2
    L_EYE_OUTER = 3
    R_EYE_INNER = 4
    R_EYE = 5
    R_EYE_OUTER = 6
    L_EAR = 7
    R_EAR = 8
    MOUTH_L = 9
    MOUTH_R = 10
    LSH = 11; RSH = 12
    LEL = 13; REL = 14
    LWR = 15; RWR = 16
    L_PINKY = 17; R_PINKY = 18
    L_INDEX = 19; R_INDEX = 20
    L_THUMB = 21; R_THUMB = 22
    LHIP = 23; RHIP = 24
    LKNE = 25; RKNE = 26
    LANK = 27; RANK = 28
    L_HEEL = 29; R_HEEL = 30
    L_FOOT = 31; R_FOOT = 32


JOINT_LABELS: List[str] = [
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
    "L_Foot",
    "R_Foot",
]


@dataclass(frozen=True)
class CameraSettings:
    COLOR_RES: Tuple[int, int] = (848, 480)
    DEPTH_RES: Tuple[int, int] = (848, 480)
    FPS: int = 30

    SPATIAL_MAGNITUDE: float = 3.0
    SPATIAL_ALPHA: float = 0.4
    SPATIAL_DELTA: float = 35.0

    TEMPORAL_ALPHA: float = 0.3
    TEMPORAL_DELTA: float = 100.0

    MIN_Z_M: float = 1.0
    MAX_Z_M: float = 3.5

    HOLE_FILLING: int = 2

    LASER_POWER: float = 300.0

    JOINT_CONFIDENCE_THRESHOLD: float = 0.7

    ENABLE_ALIGN_TO_COLOR: bool = True
    ENABLE_SPATIAL_FILTER: bool = True
    ENABLE_TEMPORAL_FILTER: bool = True
    ENABLE_HOLE_FILLING: bool = True
    ENABLE_Z_CLIP: bool = True
    ENABLE_EMITTER: bool = True


CAMERA = CameraSettings()


@dataclass(frozen=True)
class MediaPipeSettings:
    STATIC_MODE: bool = False
    MODEL_COMPLEXITY: int = 1
    SMOOTH_LANDMARKS: bool = True
    MIN_DET_CONF: float = 0.8
    MIN_TRK_CONF: float = 0.95
    VISIBILITY_THRESHOLD: float = 0.8


MP = MediaPipeSettings()


def load_config() -> Config:
    
    return Config()
