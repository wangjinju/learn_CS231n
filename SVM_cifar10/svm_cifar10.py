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
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        # 将每个分类的8个图像显示出来
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            # 创建子图像
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            # 增加标题
            if i == 0:
                plt.title(cls)
    plt.show()


def VisualizeLoss(loss_histroy):
    plt.plot(loss_histroy)
    plt.xlabel("Iteration number")
    plt.ylabel("Loss value")
    plt.show()


def PreDataset():
    cifar10_dir = 'data/cifar10'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    VisualizeImg(X_train, y_train)
    input("Enter any key to Cross-validation...")

    num_train = 49000
    num_val = 1000
    # dataset validation
    sample_index = range(num_train, num_train + num_val)
    X_val = X_train[sample_index]
    y_val = y_train[sample_index]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    # 零中心化
    X_train -= np.mean(X_train)
    X_val -= np.mean(X_val)
    X_test -= np.mean(X_test)

    # VisualizeImg(X_train, y_train) #零中心化效果显示

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    # 为偏置b在X上最后一列添加1
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    return X_train, y_train, X_val, y_val, X_test, y_test


def AutoGetPara(X_train, y_train, X_val, y_val):
    learning_rate = [1e-8, 1e-7]  # 学习率
    regularization_strengths = [5e4, 1e5]  # 正则化强度
    best_parameter = None
    best_val = -1  # 最优评分值
    delta = 1  # SVM损失函数中的delta参数
    batch_num = 200  # 小批量训练
    num_iter = 1500  # 迭代次数

    for i in learning_rate:
        for j in regularization_strengths:
            svm = SVM()
            svm.Train(X_train, y_train, j, delta, i,
                      batch_num, num_iter, True)  # 得到了权重W
            y_pred = svm.Predict(X_val)  # 利用权重W得到评分值
            acc_val = np.mean(y_val == y_pred)
            if best_val < acc_val:
                best_val = acc_val
                best_parameter = (i, j)
    print('OK! Have been identified parameter! Best validation accuracy achieved during cross-validation: %f' % best_val)
    return best_parameter


def Get_svm_model(parameter, X, y):
    svm = SVM()
    loss_histroy = svm.Train(
        X, y, parameter[1], 1, parameter[0], 200, 1500, True)
    VisualizeLoss(loss_histroy)
    input("Enter any key to predict...")
    return svm


if __name__ == "__main__":
    # 对数据进行预处理，得到训练集，测试集，验证集
    X_train, y_train, X_val, y_val, X_test, y_test = PreDataset()
    # 通过验证集自动化确定参数 learning_rate和reg
    best_parameter = AutoGetPara(X_train, y_train, X_val, y_val)
    # 通过参数和训练集构建SVM模型
    svm = Get_svm_model(best_parameter, X_train, y_train)
    # 用测试集预测准确率
    y_pred = svm.Predict(X_test)
    print("Accuracy achieved during cross-validation: %f" %
          (np.mean(y_pred == y_test)))
