import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
JOINT_LABELS = [
    "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee",
    "R_Knee", "L_Ankle", "R_Ankle", "L_Foot", "R_Foot"
]
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

CENTER=6
# INWARD = [
#     (0, 1), (0, 2), (0, 3), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 15),
#     (1, 2), (1, 3), (1, 7), (1, 8),
#     (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 14),
#     (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 14),
#     (4, 5), (4, 6), (4, 7), (4, 14),
#     (5, 6), (5, 7),
#     (6, 7), (6, 10), (6, 11), (6, 12), (6, 14),
#     (7, 8),
#     (8, 9), (8, 10), (8, 11), (8, 15),
#     (9, 10), (9, 11), (9, 15),
#     (10, 11), (10, 12), (10, 14), (10, 15),
#     (11, 12), (11, 13), (11, 14), (11, 15),
#     (12, 13), (12, 14), (12, 15),
#     (13, 14), (13, 15),
#     (14, 15)
# ]
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
OUTWARD = [(j, i) for (i, j) in INWARD]
SELF_LOOP = [(i, i) for i in range(len(ANGLE_TRIPLETS))]
G = nx.Graph()
G.add_nodes_from(range(len(ANGLE_TRIPLETS)))
G.add_edges_from(INWARD + OUTWARD+SELF_LOOP)

labels = {}
for idx, (a, b, c) in enumerate(ANGLE_TRIPLETS):
    labels[idx] = f"{idx}"

pos = nx.spring_layout(G, seed=42)

# Compute adjacency matrix
adj_matrix = nx.to_numpy_array(G)

# Create subplots: left for graph, right for matrix heatmap
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Plot graph on left
axs[0].set_title('Graph Visualization')
nx.draw_networkx_nodes(G, pos, ax=axs[0], node_size=500)
nx.draw_networkx_edges(G, pos, ax=axs[0], width=1.0)
nx.draw_networkx_labels(G, pos, labels, ax=axs[0], font_size=16)
axs[0].axis('off')

# Plot adjacency matrix heatmap on right
axs[1].set_title('Adjacency Matrix')
im = axs[1].imshow(adj_matrix, cmap='binary', interpolation='none')
axs[1].set_xticks(np.arange(16))
axs[1].set_yticks(np.arange(16))
axs[1].set_xticklabels(range(16))
axs[1].set_yticklabels(range(16))
axs[1].xaxis.tick_top()  # Move column indices to the top
fig.colorbar(im, ax=axs[1], ticks=[0, 1], shrink=0.5)

# Add text labels (1 or 0) to each cell for clarity
for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        text_color = 'black' if adj_matrix[i, j] == 0 else 'white'  # Contrast with background
        axs[1].text(j, i, str(int(adj_matrix[i, j])), va='center', ha='center', color=text_color, fontsize=8)

plt.tight_layout()
plt.show()