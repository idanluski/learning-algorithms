import numpy as np


class KMeans:
    def __init__(self, k, max_iter):
        self.k = k
        self._iter = max_iter

    def initialize(self, centroids):
        self.centroids = centroids


    def wcss(self):
        return self.WCSS

    @staticmethod
    def find_closet_distance(centroid, row):
        """

        :param centroid: 2d array each row is centruid where each column is coordinates
        :param row: represent point with num colmns coordinates
        :return: finde the closest centroid to the element and return the centroid
        """
        #dist_array = np.zeros(centroid.shape[0])
        array = (centroid - row) ** 2  # ((x_1-y_1)**2,(x_2-y_2)**2,...)
        dist_array = np.sqrt(array.sum(axis=1))
        min_dist = dist_array.min()
        idx = np.where(dist_array == min_dist)
        return idx

    def fit(self, X_train):
        self.WCSS = 0
        dict = {idx: row for idx, row in enumerate(self.centroids)} #dictionary when ID is index or cluster row and keey is row
        centroid = self.centroids
        for iter in range(self._iter):
            for row in X_train: #classify each row to cluster
                idx = KMeans.find_closet_distance(centroid, row)[0][0] #the index of the closest distance, also the ID
                up_dic = {idx: np.vstack([dict[idx], row])} #update the dictionari
                dict.update(up_dic)

            if iter == self._iter - 1:
                #print(dict)
                for key in dict:
                    cen = dict[key][0]
                    arr = (dict[key] - cen)**2
                    s = np.sum(arr, axis=1)
                    self.WCSS += np.sum(s, axis=0)
            for key in dict: #calculate the mean of each cluster
                temp = {key:np.mean(dict[key], axis=0)}
                dict.update(temp)
            centroid = np.vstack(list(dict.values())) #update the new clusters centroid

        self.centroids = centroid
        return dict



    def predict(self, X):
        array = np.zeros(X.shape[0])
        for i, elem in enumerate(X):
            array[i] = KMeans.find_closet_distance(self.centroids, elem)[0]
        return array








