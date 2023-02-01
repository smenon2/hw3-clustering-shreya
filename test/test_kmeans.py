# Write your k-means unit tests here
import numpy as np
import pytest
from cluster import *


def test_kmeans():
    # creating test set with clear centers
    test_data = np.zeros((12, 1))
    test_data[0:3, :] = 1
    test_data[3:6, :] = 2
    test_data[6:9, :] = 3
    test_data[9:12, :] = 4

    km = kmeans.KMeans(k=4)
    km.fit(test_data)
    pred = km.predict(test_data)
    centr = np.sort(km.get_centroids(), axis=None)
    known_centr = np.array([1., 2., 3., 4.])

    assert np.array_equal(centr, known_centr), "tight example data - kmeans should work"
