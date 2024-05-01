from typing import Union

from sklearn.cluster import (DBSCAN, HDBSCAN, OPTICS, AffinityPropagation,
                             AgglomerativeClustering, Birch, BisectingKMeans,
                             FeatureAgglomeration, KMeans, MeanShift,
                             MiniBatchKMeans, SpectralBiclustering,
                             SpectralClustering, SpectralCoclustering)

Clusterizer = Union[
    DBSCAN, HDBSCAN, OPTICS, AffinityPropagation,
    AgglomerativeClustering, Birch, BisectingKMeans,
    FeatureAgglomeration, KMeans, MeanShift,
    MiniBatchKMeans, SpectralBiclustering,
    SpectralClustering, SpectralCoclustering
]
