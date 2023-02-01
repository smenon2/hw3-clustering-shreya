import numpy as np
from scipy.spatial.distance import cdist
import sys


def distance(a, b):
    return np.square(np.sum((a - b) ** 2))


def update_centroids(mat, labels, centroids):
    new_cen = []
    for k in range(centroids.shape[0]):
        indices = [i for i, x in enumerate(labels) if x == k]
        new_cen.append(mat[indices].mean(axis=0))
    return np.array(new_cen)


class KMeans:
    def __init__(self, k: int, tol: float = 1e-9, max_iter: int = 100, random_seed: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.error = None
        self.centroids = None
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        np.random.seed(random_seed)

        if not isinstance(k, int):
            raise ValueError("k must be an integer")

    def _initialize_centroids(self, mat):
        """
        Initialize the centroids - K++ algorithm
        """
        k = self.k
        # randomly pick first centroid
        start_ind = np.arange(0, mat.shape[0])
        np.random.shuffle(start_ind)
        # centroids = [mat[np.random.randint(mat.shape[0]), :]]
        centroids = [mat[start_ind[0], :]]
        # now select the other centroids
        for c in range(k - 1):
            # store the distances each point in the matrix is from centroids
            dist = []
            for i in range(mat.shape[0]):
                point = mat[i, :]
                d = sys.maxsize

                for j in range(len(centroids)):
                    temp_dist = distance(point, centroids[j])
                    d = min(d, temp_dist)
                dist.append(d)

            dist = np.array(dist)
            next_centroid = mat[np.argmax(dist), :]
            centroids.append(next_centroid)
            dist = []
        return centroids

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match someß data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # np.random.seed(42)

        if self.k is None or self.k == 0:
            raise ValueError("k cannot be 0 or you have not initialized the kmeans class")

        k = self.k
        centroids = np.array(self._initialize_centroids(mat))
        print("Initial clusters:", centroids)

        err = []
        go_flag = True
        j = 0

        while go_flag:
            labels, errors = self._predict(centroids, mat)
            err.append(errors)
            centroids = update_centroids(mat, labels, centroids)

            if j > 0:
                # check to see if the error is less than the tolerance
                if err[j - 1] - err[j] <= self.tol:
                    go_flag = False
            j += 1
            if j == self.max_iter:
                break

        labels, errors = self._predict(centroids, mat)
        self.centroids = centroids
        self.error = errors

    def _predict(self, centroids, mat):

        assigned_label = []
        assign_error = []
        n = len(mat)
        k = self.k

        for obs in range(n):
            # Calculate error
            all_errors = np.array([])
            for c in range(k):
                err = (distance(centroids[c], mat[obs])**2)
                all_errors = np.append(all_errors, err)

                # Get the nearest centroid and the error
            nearest_centroid = np.where(all_errors == np.amin(all_errors))[0].tolist()[0]
            nearest_centroid_error = np.amin(all_errors)

            # Add values to corresponding lists
            assigned_label.append(nearest_centroid)
            assign_error.append(nearest_centroid_error)
        sse = np.sum(assign_error)
        print("assign", sse)

        return np.array(assigned_label), sse

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self.centroids is None:
            raise ValueError("No centroids found - please fit kmeans before trying to predict")

        labels, errors = self._predict(self.centroids, mat)
        return labels

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        return self.centroids
