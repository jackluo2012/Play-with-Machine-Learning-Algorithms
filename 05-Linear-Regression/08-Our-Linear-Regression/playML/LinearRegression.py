import numpy as np
from .metrics import r2_score

#支持多元线性方式
class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None #这个是系数 θ1-θn 之的向量
        self.intercept_ = None # 截距 对就的是 θ.
        self._theta = None #保存私有的 θ向量

    # 正规划方程来解     
    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        #用正规划方程来求解
        #np.noes 创建一个只有一行,但是有 列 为1的行
        #对于给进来的 x_train 加上一列这一列都是1
        #np.hstack 就是在横向上多加1列 
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
                      #对 求(他的逆矩阵(x_b 进行转置 去点乘 X_b)) 得到的结果 再点乘x_b 的转置)再点乘 y_train
                      # np.linalg.inv 求逆矩阵
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0] # 求出 截距
        self.coef_ = self._theta[1:] # 求出系数
        #得到线性模型的参数
        return self
    
    # 预测 结果 
    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        #先求出 X_b
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        #求出的 x_b 再点乘上 求出的细数
        return X_b.dot(self._theta)
    
    #采用 R squared 标准 评测我们的结果 
    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
