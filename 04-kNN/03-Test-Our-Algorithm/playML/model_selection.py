import numpy as np

# 训练 和 测试 数据分割 
def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"
    # 设置随机种子
    if seed: 
        np.random.seed(seed)
    #随机打乱下标
    shuffled_indexes = np.random.permutation(len(X))
    # 获取 测试数据大小
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]#测试数据集
    train_indexes = shuffled_indexes[test_size:] #训练数据集
    #训练数据集
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    #测试数据集
    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
