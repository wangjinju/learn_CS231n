import numpy as np


class SVM(object):
    def __init__(self):
        self.W = None

    '''train SVM
    X:(N,D)
    y:(N,)
    reg:(N,)
    learning_rate: gradient descent rate
    batch_num: training examples to use at each step in Mini-batch gradient descent
    num_iter: number of steps to take when optimizing
    return: loss_history
    '''

    def Train(self, X, y, reg, delta, learning_rate, batch_num, num_iter, output):
        num_train = X.shape[0]
        num_dim = X.shape[1]
        num_classes = np.max(y) + 1  # y takes values 0...K-1
        if self.W is None:
            # lazily initialize W
            self.W = 0.001*np.random.randn(num_classes, num_dim)

        # train
        loss_histroy = []
        for i in range(num_iter):
            # Mini-batch
            sample_index = np.random.choice(
                num_train, batch_num, replace=False)
            X_batch = X[sample_index, :]
            y_batch = y[sample_index]
            loss, gred = self.Svm_cost_function(X_batch, y_batch, reg, delta)
            loss_histroy.append(loss)
            self.W -= learning_rate * gred
            if output and i % 100 == 0:
                print("Iteration %d / %d:loss %f" % (i, num_iter, loss))
        return loss_histroy

    def Predict(self, X):
        scroes = X.dot(self.W.T)
        y_pred = np.zeros(X.shape[0])
        y_pred = np.argmax(scroes, axis=1)
        return y_pred

    # 核心部分，求损失函数值和对应的梯度
    def Svm_cost_function(self, X, y, reg, delta):
        '''cal loss
        :param X: A numpy array of shape (N, D)
        :param y: A numpy array of shape (N, )
        :param reg: regularization strength
        :param delta: margin
        :return: loss, gred
        '''
        num_train = X.shape[0]
        scores = X.dot(self.W.T)  # N * C，得到分值矩阵

        # 正确分类的分值，其他分值与之做比较，看差值和delta的大小
        correct_class_scores = scores[range(
            num_train), list(y)].reshape(-1, 1)  # (N,1)
        margins = scores - correct_class_scores + delta
        margins = np.maximum(0, margins)
        # do not ignore it, because 'y - y + delta' > 0, we should reset it to zeros，正确分类没有损失
        margins[range(num_train), list(y)] = 0

        loss = np.sum(margins) / num_train + 0.5 * \
            reg * np.sum(self.W * self.W)  # 0.5是为了便于求导

        # cal gred [for every example, when margin > 0, correct lable's W should -X, and wrong lable's W should +X]
        ground_true = np.zeros(margins.shape)  # N * C
        ground_true[margins > 0] = 1
        sum_margins = np.sum(ground_true, axis=1)
        ground_true[range(num_train), y] -= sum_margins

        gred = ground_true.T.dot(X) / num_train + reg * self.W
        return loss, gred
