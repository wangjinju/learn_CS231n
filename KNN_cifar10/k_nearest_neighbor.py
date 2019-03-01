import numpy as np


class KNearstNeighbor(object):
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def Train(self, X_train, y_train):
        # knn无需训练，只要存下图片数据和标签
        self.X_train = X_train
        self.y_train = y_train

    def ComDis(self, X_test):
        # 求测试集和训练集的欧氏距离（向量与向量）
        # 训练样本：X_train = [N_train * D]
        # 测试样本：X_test = [N_test * D]
        # 求出向量与向量间的欧式距离放入矩阵dists[N_test * N_train]中
        dists = np.zeros((X_test.shape[0], self.X_train.shape[0]))
        # 求两个向量欧氏距离dis，利用矩阵乘法求，利用dists = （x - y）^2求取
        value_2xy = np.multiply(X_test.dot(self.X_train.T), -2)
        value_x2 = np.sum(np.square(X_test), axis=1, keepdims=True)
        value_y2 = np.sum(np.square(self.X_train), axis=1)
        dists = value_2xy + value_x2 + value_y2  # x^2 - 2xy + y^2
        return dists

    def PredictLabel(self, dists, k):
        # 选择前k个距离最近的标签，从这些标签中选择个数最多的作为预测分类
        y_pred = np.zeros(dists.shape[0])
        for i in range(dists.shape[0]):
            # 取前K个标签
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            # 取K个标签中个数最多的标签
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred

    def Predict(self, X_test, k):
        dists = self.ComDis(X_test)
        y_pred = self.PredictLabel(dists, k)
        return y_pred
