import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap

def load_data(dest):
    path = os.path.join('work_dirs', dest, 'test_embeds.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    X = np.stack([d['pred_emb'].numpy() for d in data])
    y = np.array([d['gt_label'].item() for d in data])
    return X, y

def compute_and_save_metrics(X, y, dest, k=3, n_clusters=7):
    # 1. LOO-KNN
    dist = cosine_distances(X)
    preds = []
    for i in range(len(X)):
        nn = np.argpartition(dist[i], k+1)[1:k+1]
        preds.append(Counter(y[nn]).most_common(1)[0][0])
    knn_acc = accuracy_score(y, preds)

    # 2. Clustering
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    ari = adjusted_rand_score(y, km.labels_)
    nmi = normalized_mutual_info_score(y, km.labels_)

    # 3. Save to text
    out_dir = os.path.join('work_dirs', dest)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'results.txt')
    with open(out_path, 'w') as f:
        f.write(f'Test KNN Acc (k={k}): {knn_acc}\n')
        f.write(f'ARI: {ari}, NMI: {nmi}\n')

    print(f"Metrics saved to {out_path}")
    return preds

def plot_tsne(X, y, dest):
    tsne_proj = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
    plt.figure()
    plt.scatter(tsne_proj[:,0], tsne_proj[:,1], c=y, cmap='viridis', s=5)
    plt.colorbar(label='Exercise ID')
    plt.title('t-SNE of Embeddings')
    save_path = os.path.join('work_dirs', dest, 't-SNE.png')
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

def plot_umap(X, y, dest):
    umap_proj = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    plt.figure()
    plt.scatter(umap_proj[:,0], umap_proj[:,1], c=y, cmap='viridis', s=5)
    plt.colorbar(label='Exercise ID')
    plt.title('UMAP of Embeddings')
    save_path = os.path.join('work_dirs', dest, 'UMAP.png')
    plt.savefig(save_path)
    plt.close()
    print(f"UMAP plot saved to {save_path}")

def plot_confusion(y, preds, dest):
    cm = confusion_matrix(y, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (LOO-KNN)')
    save_path = os.path.join('work_dirs', dest, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def main():
    dest = 'suemd_4_8'
    X, y = load_data(dest)
    preds = compute_and_save_metrics(X, y, dest, k=3, n_clusters=7)
    plot_tsne(X, y, dest)
    plot_umap(X, y, dest)
    plot_confusion(y, preds, dest)

if __name__ == '__main__':
    main()


python mmaction2/tools/test.py mmaction2/configs/suemd_new_last.py    work_dirs/suemd_4_8/epoch_40.pth     --work-dir work_dirs/suemd_4_8