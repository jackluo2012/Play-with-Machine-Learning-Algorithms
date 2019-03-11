import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None # 均值 方便用户查询 这个变量 用下划线
        self.scale_ = None # 方差

    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim == 2, "The dimension of X must be 2" #只处理二维
                            #对于 x的第 i 列求均值 我们有多少个列(多少个特征我们循环就执行多少次)         
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])#均值
                            #
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])]) # 方差

        return self

    def transform(self, X):
        """将X根据这个StandardScaler进行均值方差归一化处理"""
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
               "must fit before transform!"
        assert X.shape[1] == len(self.mean_), \
               "the feature number of X must be equal to mean_ and std_"
        #赋值 为空,但是它的 shape 等于值进来的 shape
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            #对于每一列对就的 x 的值,就是原来的这一列的 x 的这一列, 减去 原始的均值 ,除以 这一列对应的标准差
            resX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col]
        return resX
