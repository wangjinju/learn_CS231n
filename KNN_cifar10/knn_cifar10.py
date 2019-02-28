from k_nearest_neighbor import KNearstNeighbor
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import numpy as np


def VisualizeImg(X_train, y_train):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 8
    for y, cls in enumerate(classes):
        # 得到该标签训练样本下标索引
        idxs = np.flatnonzero(y_train == y)
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


def CrossValidation(X_train, y_train):
    num_folds = 5  # 将训练集分成五份，以便交叉验证
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    k_accuracy = {}
    # 将数据集分为5份：训练集 = 4份训练集 + 1份验证集
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)
    # 计算每种K值
    for k in k_choices:
        k_accuracy[k] = []
        # 每个K值分别计算每份数据集作为测试集时的正确率
        for index in range(num_folds):
            # 构建1份验证集
            X_te = X_train_folds[index]
            y_te = y_train_folds[index]
            # 构建4份训练集（余下的）
            X_tr_split_others = X_train_folds[:index] + X_train_folds[index+1:]
            X_tr = X_tr_split_others[0]
            for i in range(len(X_tr_split_others) - 1):
                X_tr = np.append(X_tr, X_tr_split_others[i+1], axis=0)

            y_tr_split_others = y_train_folds[:index] + \
                y_train_folds[index + 1:]
            y_tr = y_tr_split_others[0]
            for i in range(len(y_tr_split_others) - 1):
                y_tr = np.append(y_tr, y_tr_split_others[i+1], axis=0)
            # 预测结果
            classify = KNearstNeighbor()
            classify.Train(X_tr, y_tr)
            y_te_pred = classify.Predict(X_te, k=k)
            accuracy = np.sum(y_te_pred == y_te) / float(X_te.shape[0])
            k_accuracy[k].append(accuracy)

    for k, accuracylist in k_accuracy.items():
        for accuracy in accuracylist:
            print("k = %d, accuracy = %.3f" % (k, accuracy))

    # 可视化K值效果
    for k in k_choices:
        accuracies = k_accuracy[k]
        plt.scatter([k] * len(accuracies), accuracies)
    accuracies_mean = np.array([np.mean(v)
                                for k, v in sorted(k_accuracy.items())])
    accuracies_std = np.array([np.std(v)
                               for k, v in sorted(k_accuracy.items())])
    # 根据均值和方差构建误差棒图
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


cifar10_dir = './data/cifar10'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 可视化图像
input("Enter any key to visualize dataset...")
VisualizeImg(X_train, y_train)

# 创建用于超参数调优的交叉验证集（也可以验证集，因为数据量还是很大的）
num_training = 5000
X_tr = X_train[:num_training, ::]  # [5000,32,32,3]
X_tr = np.reshape(X_tr, (X_tr.shape[0], -1))  # 将图像数据行化：[5000,3072]
y_tr = y_train[:num_training]
# print(X_tr.shape, y_tr.shape)

# 测试集500个
num_testing = 500
X_te = X_test[:num_testing, ::]
X_te = np.reshape(X_te, (X_te.shape[0], -1))
y_te = y_test[:num_testing]
# print(X_te.shape, y_te.shape)

# 交叉验证确定参数K
CrossValidation(X_tr, y_tr)
input('Enter any key to train model...')

# 训练完整数据集(这里就以5000个数据集作为完整训练集，500个数据集作为测试集(60000个数据电脑内存吃不消), k值根据图显示10应为最佳)
classify = KNearstNeighbor()
classify.Train(X_tr, y_tr)
y_te_pred = classify.Predict(X_te, k=10)
accuracy = np.sum(y_te_pred == y_te) / float(X_te.shape[0])
print('最终测试： '
      '     K = %d, accuracy = %.3f' % (10, accuracy))
