from __future__ import annotations
import numpy as np
from mmaction.models.utils.graph import Graph as BaseGraph, edge2mat, normalize_digraph
from mmaction.registry import MODELS

JOINT_LABELS = [
    "Head", "L_Shoulder", "R_Shoulder", "L_Elbow",  "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Hip", "R_Hip",     "L_Knee",     "R_Knee",   "L_Ankle", "R_Ankle", "L_Foot", "R_Foot",
]
ROW = {n: i for i, n in enumerate(JOINT_LABELS)}

ANGLE_TRIPLETS = np.array([
    (ROW["R_Shoulder"], ROW["L_Shoulder"], ROW["L_Elbow"]),
    (ROW["L_Shoulder"], ROW["L_Elbow"],    ROW["L_Wrist"]),
    (ROW["L_Elbow"],    ROW["L_Shoulder"], ROW["L_Hip"]),
    (ROW["L_Shoulder"], ROW["L_Hip"],      ROW["L_Knee"]),
    (ROW["L_Hip"],      ROW["L_Knee"],     ROW["L_Ankle"]),
    (ROW["L_Knee"],     ROW["L_Ankle"],    ROW["L_Foot"]),
    (ROW["L_Knee"],     ROW["L_Hip"],      ROW["R_Hip"]),
    (ROW["L_Foot"],     ROW["L_Knee"],     ROW["L_Shoulder"]),
    (ROW["L_Shoulder"], ROW["R_Shoulder"], ROW["R_Elbow"]),
    (ROW["R_Shoulder"], ROW["R_Elbow"],    ROW["R_Wrist"]),
    (ROW["R_Elbow"],    ROW["R_Shoulder"], ROW["R_Hip"]),
    (ROW["R_Shoulder"], ROW["R_Hip"],      ROW["R_Knee"]),
    (ROW["R_Hip"],      ROW["R_Knee"],     ROW["R_Ankle"]),
    (ROW["R_Knee"],     ROW["R_Ankle"],    ROW["R_Foot"]),
    (ROW["R_Knee"],     ROW["R_Hip"],      ROW["L_Hip"]),
    (ROW["R_Foot"],     ROW["R_Knee"],     ROW["R_Shoulder"]),
], dtype=int)

NUM_NODE = ANGLE_TRIPLETS.shape[0]
CENTER    = 6

INWARD = [
    # Left chain (nodes 0-1-2-3-4-5-7)
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7),
    # Right chain (nodes 8-9-10-11-12-13-15)
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 15),
    # Hip cross for node 6
    (3,6), (6,11), (4,6), (12,6), (5,6), (13,6),
    # Hip cross for node 14 (symmetric)
    (11,14), (14,3), (12,14), (4,14), (13,14), (5,14),
    # Shoulder cross
    (0,8), (1,9), (2,10),
    # Foot to shoulder closes
    (7,0), (15,8)
]

OUTWARD   = [(j, i) for (i, j) in INWARD]
SELF_LOOP = [(i, i) for i in range(NUM_NODE)]


def build_adjacency(num_node: int, self_link, inward, outward) -> np.ndarray:
    
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    return np.stack((I, In, Out))


@MODELS.register_module(name='angles16_kinematic')
class AngleGraph16Kinematic(BaseGraph):
    

    def __init__(self, layout: str = 'angles16_kinematic', mode: str = 'spatial'):
        A = build_adjacency(NUM_NODE, SELF_LOOP, INWARD, OUTWARD)
        super().__init__(
            layout=dict(num_node=NUM_NODE, inward=INWARD, center=CENTER),
            mode=mode)
        self.A = A
