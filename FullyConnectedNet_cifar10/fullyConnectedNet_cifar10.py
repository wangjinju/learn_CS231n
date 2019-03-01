from fullyConnectedNet import FullyConnectedNet
from solver import Solver
from data_utils import load_CIFAR10
import numpy as np
import matplotlib.pyplot as plt


def auto_get_params():
    pass


def create_best_model():
    pass


def compare_batchnorm():
    pass


def compare_dropout():
    pass


# 比较不同的优化方法
def compare_optims(data):
    num_train = 4000  # 取4000个训练样本
    small_data = {
        'X_train': data['X_train'][:num_train],  # 取训练集的前4000个
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],  # 取所有的验证集
        'y_val': data['y_val']
    }

    solvers = {}
    input_dims = 32 * 32 * 3
    hidden_dims = [100, 100, 100, 100, 100]  # 隐藏层5个神经元
    num_classes = 10
    weight_scale = 5e-2
    learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3,
                      'sgd': 1e-2, 'sgd_momentum': 1e-2}  # 不同的最优化方法
    reg = 0.0  # 正则化（regularization）惩罚
    update_rule = ['rmsprop', 'sgd_momentum', 'sgd', 'adam']  # 权重更新方法

    for uo in update_rule:
        model = FullyConnectedNet(
            input_dims, hidden_dims, num_classes, weight_scale, reg)  # 构建两层神经网络

    pass


def pre_dataset(path):
    X_train, y_train, X_test, y_test = load_CIFAR10(path)
    num_train = 49000
    num_val = 1000
    # 将训练集分为训练集和验证集，49:1
    mask = range(num_train, num_train + num_val)
    X_val = X_train[mask]
    y_val = y_train[mask]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    # 数据零中心化
    X_train -= np.mean(X_train)
    X_val -= np.mean(X_val)
    X_test -= np.mean(X_test)

    # 将维度展开
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # # 打印数据维度
    # print('Train data shape: {}'.format(X_train.shape))
    # print('Train labels shape: {}'.format(y_train.shape))
    # print('Validation data shape: {}'.format(X_val.shape))
    # print('Validation labels shape: {}'.format(y_val.shape))
    # print('Test data shape: {}'.format(X_test.shape))
    # print('Test labels shape: {}'.format(y_test.shape))

    # 建立数据字典，无序性，不分先后
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    return data


if __name__ == "__main__":
    path = 'data/cifar10'
    data = pre_dataset(path)
    compare_optims(data)
