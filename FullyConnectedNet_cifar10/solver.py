from fullyConnectedNet import FullyConnectedNet
import numpy as np
import matplotlib.pyplot as plt
import optim


class Solver(object):
    '''  该类用于接收数据与标签，对权值进行相应求解，在solver类中调整一些超参数以达到最好的训练效果
    solver = Solver(model, data,
                optim_config={
                    'learning_rate': learning_rates[uo]
                },
                batch_size=100,
                iters_per_ann=400,
                num_epochs=5,
                update_rule=uo,
                print_every=100,
                verbose=True,
                lr_decay=1
                )      
    Optional arguments:
    - update_rule: A string giving the name of an update rule in optim.py.
      Default is 'sgd'.
    - optim_config: A dictionary containing hyperparameters that will be
      passed to the chosen update rule. Each update rule requires different
      hyperparameters (see optim.py) but all update rules require a
      'learning_rate' parameter so that should always be present.
    - lr_decay: A scalar for learning rate decay; after each epoch the learning
      rate is multiplied by this value.
    - batch_size: Size of minibatches used to compute loss and gradient during
      training.
    - num_epochs: The number of epochs to run for during training.
    - print_every: Integer; training losses will be printed every print_every
      iterations.
    - verbose: Boolean; if set to false then no output will be printed during
      training.
    '''
    # **kwargs用作传递键值可变长参数列表

    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # unpack keyword arguments
        self.batch_size = kwargs.pop('batch_size', 100)  # 小批量训练
        self.iters_per_ann = kwargs.pop(
            'iters_per_ann', 100)  # 每迭代这么多次，学习率改变一次
        self.num_epochs = kwargs.pop('num_epochs', 10)  # 横坐标：10个完整周期
        self.update_rule = kwargs.pop('update_rule', 'sgd')  # 最优化方法:默认sgd
        # 不同的最优化方法超参数设置，默认均包含学习率0.0001
        self.optim_config = kwargs.pop('optim_config', {})
        self.verbose = kwargs.pop('verbose', True)  # 是否打印出结果
        self.print_every = kwargs.pop('print_every', 10)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)  # 学习率退火

        # Make sure the update rule exists, then replace the string
        # name with the actual function，代替optim中对应的函数
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule  "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        # Make a deep copy of the optim_config for each parameter
        self._reset()

    def _reset(self):
        """
        重置函数对一些solver类中的变量进行了重置。
        特别注意的是新建了一个
        optim_configs字典来存储优化的参数，
        之前的优化参数保存在self.optim_config字典中，这两个是完全不一样的！！
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        '''
        Make a single gradient update. This is called by train() and should not
        be called manually.
        小批量训练梯度更新
        先求损失值和梯度，再求参数更新
        '''
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(
            num_train, self.batch_size, replace=True)  # 小批次常用操作，掩膜
        X_batch = self.X_train[batch_mask]  # 从训练集中抽取
        y_batch = self.y_train[batch_mask]  # 对应标志

        # 计算损失函数和梯度
        loss, grads = self.model.loss(X_batch, y_batch)  # 调用模型loss函数计算
        self.loss_history.append(loss)  # 画损失值曲线图

        # Perform a parameter update
        for p, w in self.model.params.items():  # P,w对应键值
            dw = grads[p]
            config = self.optim_configs[p]  # 优化后的超参数
            next_w, next_config = self.update_rule(
                w, dw, config)  # 执行optim中的最优化函数
            self.model.params[p], self.optim_configs[p] = next_w, next_config

    def check_accuracy(self, X, y, num_samples=None):
        '''Check accuracy of the model on the provided data.
        - num_samples: If not None, subsample the data and only test the model
        on num_samples datapoints.取数据的一部分
        - self.batch_size: Split X and y into batches of this size to avoid using too
        much memory.将数据一部分分批次
        return:
        - acc: Scalar giving the fraction of instances that were correctly
        classified by the model.
        '''
        N = X.shape[0]
        if num_samples is not None:
            mask = np.random.choice(N, num_samples, replace=True)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batchsize = int(N/self.batch_size)
        if N % self.batch_size != 0:  # 除不尽加1
            num_batchsize += 1

        y_pred = []
        for i in range(num_batchsize):
            start = int(i*self.batch_size)
            end = int((i+1)*self.batch_size)
            scroes = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scroes, axis=1))
        y_pred = np.hstack(y_pred)
        acc = (y_pred == y).mean()
        return acc

    def train(self):
        '''Run optimization to train the model.
        '''
        num_train = self.X_train.shape[0]
        # 训练完所有批次需要的次数（训练完所有批次定义为一个周期）
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iters = int(self.num_epochs * iterations_per_epoch)  # 总的小批量次数
        epoch = 0
        for it in range(num_iters):  # 循环训练小样本，一次100个
            # 更新一下。每次更新都是从所有例子中，抽取batch_size个例子，
            # 所以batch越小，要想覆盖所有的数据集
            # 所需要的迭代次数越多，也就解释了上面的iterations_per_epoch的来源
            self._step()

            # 学习率退火
            if (it+1) % self.iters_per_ann == 0:
                for c in self.optim_configs:
                    self.optim_configs[c]['learning_rate'] *= self.lr_decay

            # 每迭代print_every次，打印出一次结果
            if self.verbose and (it % self.print_every == 0):
                print('(Iteration %d / %d) loss: %f'
                      % (it+1, num_iters, self.loss_history[-1]))

            # 经过一个周期，或者训练完毕
            if self.verbose and (it % iterations_per_epoch or it == (num_iters - 1)) == 0:
                epoch += 1
                # 每隔一个周期，求测试集上的精度，便于调参
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=1000)
                self.train_acc_history.append(train_acc)
                # 验证集上的精度，正确的情况下，应和测试集上的精度差不多
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.val_acc_history.append(val_acc)
                print('(Epoch %d / %d) train acc: %f; val_acc: %f' %
                      (epoch, self.num_epochs, train_acc, val_acc))

    def visualization_model(self):
        pass
