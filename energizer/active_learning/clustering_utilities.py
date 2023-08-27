import numpy as np
from numpy.random import RandomState
from typing import List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist


def _similarity(centers: np.ndarray, X: np.ndarray, normalized: bool) -> np.ndarray:
    sim = np.matmul(centers, X.T)
    if not normalized:
        sim = sim / np.dot(np.linalg.norm(centers, axis=1)[:, np.newaxis], np.linalg.norm(X, axis=1)[np.newaxis, :])
    return sim


def _get_nearest_to_centers_batch(centers: np.ndarray, X: np.ndarray, normalized: bool) -> List[int]:
    return _similarity(centers, X, normalized).argmax(axis=1).tolist()


def _get_nearest_to_centers_iterative(centers: np.ndarray, X: np.ndarray, normalized: bool) -> List[int]:
    indices = np.empty(centers.shape[0], dtype=int)
    for i in range(centers.shape[0]):
        sim = _similarity(centers[None, i], X, normalized)
        sim[0, indices[0:i]] = -np.inf
        indices[i] = sim.argmax()

    return indices.tolist()


def _get_nearest_to_centers(centers: np.ndarray, X: np.ndarray, normalized: bool, num_clusters: int) -> List[int]:
    indices = _get_nearest_to_centers_batch(centers, X, normalized)

    # fall back to an iterative version if one or more vectors are most similar
    # to multiple cluster centers
    if np.unique(indices).shape[0] < num_clusters:
        indices = _get_nearest_to_centers_iterative(centers, X, normalized)

    return indices


def _silhouette_k_select(X: np.ndarray, max_k: int, rng: RandomState) -> int:

    silhouette_avg_n_clusters = []
    k_options = list(range(2, max_k))
    for n_clusters in k_options:

        # Initialize the clusterer with n_clusters value and a random generator
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=rng)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(X, cluster_labels, random_state=rng)
        silhouette_avg_n_clusters.append(silhouette_avg)

    return k_options[np.argmax(silhouette_avg_n_clusters).item()]


def kmeans(
    X: np.ndarray, num_clusters: int, rng: RandomState, use_silhouette: bool = False, normalize: bool = True
) -> List[int]:

    if normalize:
        X = StandardScaler().fit_transform(X)

    num_clusters = min(X.shape[0], num_clusters)
    if num_clusters > 1 and use_silhouette:
        num_clusters = _silhouette_k_select(X, max_k=num_clusters, rng=rng)

    cluster_learner = KMeans(n_clusters=num_clusters, n_init="auto", random_state=rng)
    cluster_learner.fit(X)
    centers = cluster_learner.cluster_centers_

    return _get_nearest_to_centers(centers, X, normalize, num_clusters)


def kmeans_pp_sampling(X: np.ndarray, k: int) -> List[int]:
    """kmeans++ algorithm used for sampling as an alternative to the determinantal point process."""

    # randomly choose first center
    centers_ids = [np.random.choice(X.shape[0])]

    # greedily choose centers
    for _ in range(k - 1):
        dist = cdist(X, X[centers_ids, :]).min(axis=1) ** 2
        prob = dist / dist.sum()
        new_center_id = np.random.choice(X.shape[0], p=prob)
        centers_ids.append(new_center_id)

    return centers_ids