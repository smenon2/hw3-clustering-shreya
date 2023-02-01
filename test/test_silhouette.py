# write your silhouette score unit tests here
from cluster import *
import pytest
import sklearn
from sklearn.metrics import silhouette_samples
import numpy as np


def test_silhouette_score():
    clusters, labels = utils.make_clusters(k=4, scale=1)
    km = kmeans.KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    scores_sk = silhouette_samples(clusters, pred)

    scores = np.ndarray.round(np.array(scores), decimals=3, out=None)
    scores_sk = np.ndarray.round(np.array(scores_sk), decimals=3, out=None)
    assert np.array_equal(scores, scores_sk), "Silhouette scores should be the same as sklearn "
