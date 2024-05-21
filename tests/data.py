CONFIG_DATA = {
    "clustering": {
        "type": "kmeans",
        "kmeans": {"n_clusters": 8},
        "spectral": {"n_clusters": 8, "affinity": "nearest_neighbors"},
        "dbscan": {"eps": 0.5, "min_samples": 5},
    }
}
