import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from svm import SVM


def VisualizeImg(X_train, y_train):
    """可视化
    :X_train: 训练集
    :y_train: 训练标签    
    """
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams['image.interpolation'] = 'nearest'  # 最近邻插值
    plt.rcParams['image.cmap'] = 'gray'  # 灰度映射

    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 8
    for y, cls in enumerate(classes):  # 将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
        # y_train == y返回0/1向量，flatnonzero返回非0元素位置
        idxs = np.flatnonzero(y_train == y)  # 得到该标签训练样本下标索引
        # 从某一分类的下标中随机选择8个图像（replace设为False确保不会选择到同一个图像）

    pass
