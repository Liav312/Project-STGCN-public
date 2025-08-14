# import torch
# import torch.nn.functional as F

# def multi_similarity_loss(embeddings, labels, alpha=2.0, beta=50.0, lambd=0.5, epsilon=0.1, mining_strategy='semi_hard_to_hard', epoch=None, total_epochs=40, distance='cosine'): 
#     B = embeddings.size(0)
#     if distance == 'cosine':
#         sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
#     elif distance == 'euclidean':
#         dist = torch.cdist(embeddings, embeddings)
#         sim = -dist  # Convert to similarity (higher better); normalize if needed: sim = 1 / (1 + dist)
#     # Mining
#     pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
#     neg_mask = ~pos_mask
#     pos_mask.fill_diagonal_(0)  # Exclude self

#     if mining_strategy == 'semi_hard_to_hard':
#         thresh = 0.8 * total_epochs  # Switch at 80%
#         if epoch < thresh:
#             # Semi-hard: Select pos > min_pos + eps, neg < max_neg - eps
#             min_pos = sim[pos_mask].min() if pos_mask.any() else 0
#             max_neg = sim[neg_mask].max() if neg_mask.any() else 0
#             pos_select = sim > min_pos + epsilon
#             neg_select = sim < max_neg - epsilon
#         else:
#             # Hard: All, or hardest (min pos, max neg)
#             pos_select = pos_mask
#             neg_select = neg_mask
#     else:
#         pos_select = pos_mask
#         neg_select = neg_mask

#     # Loss
#     # Start from a zero tensor that keeps the computation graph so that
#     # ``loss.backward()`` will not fail even when no valid pairs exist in the
#     # current batch.
#     loss = embeddings.sum() * 0
#     for i in range(B):
#         pos_sim = sim[i][pos_select[i]]
#         neg_sim = sim[i][neg_select[i]]
#         if len(pos_sim) > 0:
#             loss += (1 / alpha) * F.log_softmax(alpha * (pos_sim - lambd), dim=0).neg().mean()
#         if len(neg_sim) > 0:
#             loss += (1 / beta) * F.log_softmax(beta * (lambd - neg_sim), dim=0).neg().mean()
#     return loss / B

import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, distances

def multi_similarity_loss(embeddings, labels, alpha=2.0, beta=50.0, lambd=0.5, epsilon=0.1, distance='cosine'):
    # Note: Ignoring custom mining_strategy, epoch, total_epochs to use original PyTorch Metric Learning impl
    # Install via: pip install pytorch-metric-learning

    # Select distance metric
    if distance == 'cosine':
        dist_func = distances.CosineSimilarity()
    elif distance == 'euclidean':
        dist_func = distances.LpDistance(p=2, normalize_embeddings=True)
    else:
        raise ValueError(f"Unsupported distance: {distance}")

    # Original miner and loss
    miner = miners.MultiSimilarityMiner(epsilon=epsilon, distance=dist_func)
    loss_func = losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=lambd, distance=dist_func)

    # Mine hard pairs
    hard_pairs = miner(embeddings, labels)

    # Compute loss
    return loss_func(embeddings, labels, hard_pairs)