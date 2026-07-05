"""SKILL #2 —— 模型与训练(「用什么模型 / 怎么训练」)。
在(已被 feature_engineer 处理过的)训练数据上拟合一个模型,给训练集和测试集都输出概率。

训练到位的 logistic:更多迭代 + 合适学习率 + L2 正则(防止在 27 维展开特征上过拟合)。"""
import numpy as np


def train_predict(train_X, train_y, test_X, iters=3000, lr=0.1, l2=0.1):
    Xtr = np.asarray(train_X, float)
    y = np.asarray(train_y, float)
    Xte = np.asarray(test_X, float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    Xte = np.hstack([np.ones((len(Xte), 1)), Xte])
    w = np.zeros(Xtr.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xtr @ w))
        grad = (Xtr.T @ (p - y)) / len(y)
        grad[1:] += l2 * w[1:] / len(y)  # 不惩罚截距项
        w -= lr * grad
    return {
        "train": [float(v) for v in 1 / (1 + np.exp(-Xtr @ w))],
        "test": [float(v) for v in 1 / (1 + np.exp(-Xte @ w))],
    }
