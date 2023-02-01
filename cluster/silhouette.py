import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        # first get the number of samples and clusters
        number_samples = X.shape[0]
        number_clusters = len(np.unique(y))

        # initialize a and b arrays
        a = np.zeros(number_samples)
        b = np.zeros(number_samples)
        ss = []

        for i in range(number_samples):

            # get the labels and data points for all the other points
            labels_others = y[np.arange(number_samples) != i]
            data_others = X[np.arange(number_samples) != i]

            # Find the mean distance of that point to all the points in the same cluster
            a[i] = np.mean([np.linalg.norm(X[i] - data_others[j]) for j in np.where(labels_others == y[i])[0]])

            # Find the mean distance of that point to the nearest cluster
            b_values = []
            for c in np.unique(labels_others[labels_others != y[i]]):
                val = []
                for j in np.where(labels_others == c)[0]:
                    val.append(np.linalg.norm(X[i] - data_others[j]))
                b_values.append(np.mean(val))

            b[i] = np.min(b_values) if b_values else 0

        for i in range(number_samples):
            ss.append((b[i] - a[i]) / max(a[i], b[i]))

        return ss

