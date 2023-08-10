import numpy as np


class KNN:
    def __init__(self, k) -> None:
        if isinstance(k, int):
            self.k = k
        else:
            raise Exception("error with input")

    def fit(self, x_train, y_train) -> None:
        self.x_train = x_train
        self.y_train = y_train


    def predict_each_row(self, x_test): #x_text here is 1-d array
        array = (self.x_train - x_test) ** 2  # ((x_1-y_1)**2,(x_2-y_2)**2,...)
        dist_array = np.sqrt(array.sum(axis=1))
        idx = np.argpartition(dist_array, self.k)  # get index of K'th nighbors
        K_lable = self.y_train[idx[:self.k]]  # get the lable of K'th nighbors
        text_lable_count = np.bincount(K_lable)  # find the most frequens lable of the naighbors
        max_lable = np.max(text_lable_count)
        clssiffy_lable = np.where(text_lable_count == max_lable)
        return np.random.choice(clssiffy_lable[0])


    def predict(self, x_test):
        lable_array = np.zeros(x_test.shape[0])
        for row in range(x_test.shape[0]):
            lable_array[row] = KNN.predict_each_row(self, x_test[row])
        print(lable_array)
        return lable_array




