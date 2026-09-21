"""
Customer Segmentation with K-Means
----------------------------------
Unsupervised clustering: group synthetic customer data into segments based
on annual income and spending score, then describe each segment.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def make_customers(n=300, seed=42):
    """Generate synthetic customers: income (k$) and spending score (1-100)."""
    rng = np.random.default_rng(seed)
    centers = [(30, 20), (30, 80), (60, 50), (90, 20), (90, 80)]
    data = []
    for cx, cy in centers:
        pts = rng.normal(loc=(cx, cy), scale=(6, 8), size=(n // len(centers), 2))
        data.append(pts)
    return np.vstack(data)


def main():
    X = make_customers()
    X_scaled = StandardScaler().fit_transform(X)

    k = 5
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    print(f"Clusters: {k}")
    print(f"Silhouette score: {sil:.3f}\n")

    print("Segment profiles (avg income k$, avg spending score):")
    for c in range(k):
        members = X[labels == c]
        print(
            f"  Segment {c}: income={members[:,0].mean():5.1f}  "
            f"spending={members[:,1].mean():5.1f}  size={len(members)}"
        )


if __name__ == "__main__":
    main()
