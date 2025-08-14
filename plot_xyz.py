# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
# # ---------- CONFIG -----------------------------------------------------------
# NPY_PATH = "data/raw/suemd-markless/S4A4D3R4.npy"          # ← change to your file
# JOINT_LABELS = [
#     "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist",
#     "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
#     "L_Foot", "R_Foot",
# ]
# # ------------------------------------------------------------------------------
# ROW_MP15 = {name: i for i, name in enumerate(JOINT_LABELS)}
# _ANG_TRIPLETS = [
#     (ROW_MP15["R_Shoulder"], ROW_MP15["L_Shoulder"], ROW_MP15["L_Elbow"]),
#     (ROW_MP15["L_Shoulder"], ROW_MP15["L_Elbow"], ROW_MP15["L_Wrist"]),
#     (ROW_MP15["L_Elbow"],    ROW_MP15["L_Shoulder"], ROW_MP15["L_Hip"]),
#     (ROW_MP15["L_Shoulder"], ROW_MP15["L_Hip"], ROW_MP15["L_Knee"]),
#     (ROW_MP15["L_Hip"],      ROW_MP15["L_Knee"], ROW_MP15["L_Ankle"]),
#     (ROW_MP15["L_Knee"],     ROW_MP15["L_Ankle"], ROW_MP15["L_Foot"]),
#     (ROW_MP15["L_Knee"],     ROW_MP15["L_Hip"], ROW_MP15["R_Hip"]),
#     (ROW_MP15["L_Foot"],     ROW_MP15["L_Knee"], ROW_MP15["L_Shoulder"]),
#     (ROW_MP15["L_Shoulder"], ROW_MP15["R_Shoulder"], ROW_MP15["R_Elbow"]),
#     (ROW_MP15["R_Shoulder"], ROW_MP15["R_Elbow"], ROW_MP15["R_Wrist"]),
#     (ROW_MP15["R_Elbow"],    ROW_MP15["R_Shoulder"], ROW_MP15["R_Hip"]),
#     (ROW_MP15["R_Shoulder"], ROW_MP15["R_Hip"], ROW_MP15["R_Knee"]),
#     (ROW_MP15["R_Hip"],      ROW_MP15["R_Knee"], ROW_MP15["R_Ankle"]),
#     (ROW_MP15["R_Knee"],     ROW_MP15["R_Ankle"], ROW_MP15["R_Foot"]),
#     (ROW_MP15["R_Knee"],     ROW_MP15["R_Hip"], ROW_MP15["L_Hip"]),
#     (ROW_MP15["R_Foot"],     ROW_MP15["R_Knee"], ROW_MP15["R_Shoulder"]),
# ]

# ANGLE_LABELS = [
#     "L_Shoulder-R_Shoulder-L_Elbow",
#     "L_Shoulder-L_Elbow-L_Wrist",
#     "L_Elbow-L_Shoulder-L_Hip",
#     "L_Shoulder-L_Hip-L_Knee",
#     "L_Hip-L_Knee-L_Ankle",
#     "L_Knee-L_Ankle-L_Foot",
#     "L_Knee-L_Hip-R_Hip",
#     "L_Foot-L_Knee-L_Shoulder",
#     "R_Shoulder-L_Shoulder-R_Elbow",
#     "R_Shoulder-R_Elbow-R_Wrist",
#     "R_Elbow-R_Shoulder-R_Hip",
#     "R_Shoulder-R_Hip-R_Knee",
#     "R_Hip-R_Knee-R_Ankle",
#     "R_Knee-R_Ankle-R_Foot",
#     "R_Knee-R_Hip-L_Hip",
#     "R_Foot-R_Knee-R_Shoulder",
# ]
# _ANG_TRIPLETS = np.array(_ANG_TRIPLETS, dtype=int)
# import numpy as np

# JOINT_LABELS = [
#     "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist",
#     "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
#     "L_Foot", "R_Foot",
# ]

# ROW_MP15 = {name: i for i, name in enumerate(JOINT_LABELS)}

# _ANG_TRIPLETS = [
#     (ROW_MP15["R_Shoulder"], ROW_MP15["L_Shoulder"], ROW_MP15["L_Elbow"]),
#     (ROW_MP15["L_Shoulder"], ROW_MP15["L_Elbow"], ROW_MP15["L_Wrist"]),
#     (ROW_MP15["L_Elbow"],    ROW_MP15["L_Shoulder"], ROW_MP15["L_Hip"]),
#     (ROW_MP15["L_Shoulder"], ROW_MP15["L_Hip"], ROW_MP15["L_Knee"]),
#     (ROW_MP15["L_Hip"],      ROW_MP15["L_Knee"], ROW_MP15["L_Ankle"]),
#     (ROW_MP15["L_Knee"],     ROW_MP15["L_Ankle"], ROW_MP15["L_Foot"]),
#     (ROW_MP15["L_Knee"],     ROW_MP15["L_Hip"], ROW_MP15["R_Hip"]),
#     (ROW_MP15["L_Foot"],     ROW_MP15["L_Knee"], ROW_MP15["L_Shoulder"]),
#     (ROW_MP15["L_Shoulder"], ROW_MP15["R_Shoulder"], ROW_MP15["R_Elbow"]),
#     (ROW_MP15["R_Shoulder"], ROW_MP15["R_Elbow"], ROW_MP15["R_Wrist"]),
#     (ROW_MP15["R_Elbow"],    ROW_MP15["R_Shoulder"], ROW_MP15["R_Hip"]),
#     (ROW_MP15["R_Shoulder"], ROW_MP15["R_Hip"], ROW_MP15["R_Knee"]),
#     (ROW_MP15["R_Hip"],      ROW_MP15["R_Knee"], ROW_MP15["R_Ankle"]),
#     (ROW_MP15["R_Knee"],     ROW_MP15["R_Ankle"], ROW_MP15["R_Foot"]),
#     (ROW_MP15["R_Knee"],     ROW_MP15["R_Hip"], ROW_MP15["L_Hip"]),
#     (ROW_MP15["R_Foot"],     ROW_MP15["R_Knee"], ROW_MP15["R_Shoulder"]),

# ]

# _ANG_TRIPLETS = np.array(_ANG_TRIPLETS, dtype=int)


# def compute_angles_mp15(coords: np.ndarray) -> np.ndarray:
#     if coords.ndim != 3 or coords.shape[1] < len(JOINT_LABELS):
#         raise ValueError("coords must have shape (T, 15, 3)")

#     pA = coords[:, _ANG_TRIPLETS[:, 0]]
#     pB = coords[:, _ANG_TRIPLETS[:, 1]]
#     pC = coords[:, _ANG_TRIPLETS[:, 2]]

#     v1 = pA - pB
#     v2 = pC - pB
#     dot = np.einsum("tij,tij->ti", v1, v2)
#     norm = np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2)

#     cos = dot / (norm + 1e-8)
#     cos = np.clip(cos, -1.0, 1.0)
#     angles = np.degrees(np.arccos(cos))
#     angles = np.clip(angles, 1.0, 179.0)
#     return angles.astype(np.float32)


# data = np.load(NPY_PATH)               # shape expected (frames, joints, 3)
# print(data.shape)

# xy =savgol_filter(data[:, :, 0:2], 9,3 , axis=0)  
# #xy=data[:,:,0:2]
# z = savgol_filter(data[:, :, 2:3], 9,2 , axis=0)

# data= np.concatenate((xy, z), axis=2)

# if data.ndim == 2 and data.shape[1] == len(JOINT_LABELS) * 3:
#     # file is flattened (frames, 3*joints) → reshape
#     data = data.reshape(data.shape[0], len(JOINT_LABELS), 3)

# assert data.shape[1] == len(JOINT_LABELS), (
#     f"{data.shape[1]} joints in file, but {len(JOINT_LABELS)} labels supplied."
# )


# ang_full=compute_angles_mp15(data)
# #ang_full = np.load("data/angles-old/suemd-markless/S4A4D3R4.npy")
# #ang_full1=compute_isb_angles(data)
# frames = np.arange(data.shape[0])
# components = ["X", "Y", "Z"]
# # fig2 = plt.figure(figsize=(20, 12))

# # for j in range(ang_full.shape[1]):
# #     plt.plot(ang_full1[:, j], lw=1, label=ANGLE_LABELS[j])


# fig3 = plt.figure(figsize=(20, 12))

# for j in range(ang_full.shape[1]):
#     plt.plot(ang_full[:, j], lw=1, label=ANGLE_LABELS[j])


# fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

# for c_idx, ax in enumerate(axes):
#     for j_idx, label in enumerate(JOINT_LABELS):
#         ax.plot(frames, data[:, j_idx, c_idx], label=label)
#     ax.set_ylabel(f"{components[c_idx]} coordinate")
#     ax.set_title(f"{components[c_idx]} vs. Frame")
#     ax.grid(True, alpha=0.3)
#     # Keep the legend tidy but complete
#     ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")

# axes[-1].set_xlabel("Frame #")
# plt.tight_layout()
# plt.show()
import pickle

ann_path = 'data/split/suemd-markless/suemd_train_windows.pkl'
lengths_path = 'data/split/suemd-markless/suemd_clip_lengths_train.pkl'

with open(ann_path, 'rb') as f:
    anns = pickle.load(f)

with open(lengths_path, 'rb') as f:
    lengths = pickle.load(f)

window = 50  # From config

frame_dirs_ann = set(ann['frame_dir'] for ann in anns)
frame_dirs_len = set(lengths.keys()) if isinstance(lengths, dict) else set()

missing_dirs = frame_dirs_ann - frame_dirs_len
print(f"Unique frame_dir in anns: {len(frame_dirs_ann)}")
print(f"Unique frame_dir in lengths: {len(frame_dirs_len)}")
print(f"Missing lengths for {len(missing_dirs)} frame_dirs")
if missing_dirs:
    print(f"Examples: {list(missing_dirs)[:5]}")

invalid_windows = [ann for ann in anns if ann.get('i0', 0) + window > lengths.get(ann['frame_dir'], 0)]
print(f"Invalid windows: {len(invalid_windows)} / {len(anns)}")
if invalid_windows:
    print(f"Example invalid: {invalid_windows[0]}")