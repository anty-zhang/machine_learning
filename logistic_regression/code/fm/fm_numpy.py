# -*- coding: utf-8 -*-
import numpy as np
import time


# 定义sigmoid函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 结果预测
def prediction(dataMatrix, w0, w, v):
    m = np.shape(dataMatrix)[0]
    result_list = []

    for i in range(m):
        v_1 = dataMatrix[i] * v
        v_2 = np.multiply(dataMatrix[i], dataMatrix[i]) * np.multiply(v, v)
        interaction = 0.5 * np.sum(np.multiply(v_1, v_1) - v_2)
        predict = w0 + dataMatrix[i] * w + interaction
        result_list.append(sigmoid(predict[0, 0]))

    return result_list


def get_accuracy(predict, y):
    m = np.shape(predict)[0]

    acc = 0
    for i in range(m):
        if float(predict[i]) >= 0.5 and y[i] == 1.0:
            acc += 1

    return float(acc) / float(m+1)


def save_mode(model_file_name, w0, w, v):
    with open(model_file_name, 'w') as f:
        f.write(str(w0) + '\n')

        w_list = []
        for i in range(np.shape(w)[0]):
            w_list.append(str(w[i, 0]))
        f.write(",".join(w_list) + "\n")

        m, n = np.shape(v)
        for i in range(m):
            v_list = []
            for j in range(n):
                v_list.append(str(v[i, j]))
            f.write(",".join(v_list) + "\n")


# load train data
def load_data(filename):
    feature_list = []
    label_list = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split("\t")
            item_feature_list = []
            for i in range(len(fields) - 1):
                item_feature_list.append(float(fields[i]))

            feature_list.append(item_feature_list)
            label_list.append(int(fields[len(fields) - 1]))
    print("FEATURE LEN: ", len(feature_list))
    return feature_list, label_list


# init weight
# w: 初始化权重
# v: 交叉项权重
# n,k: v特征维度
def initialize_w_v(n, k):
    w = np.ones((n, 1))
    v = np.mat(np.zeros((n, k)))

    for i in range(n):
        for j in range(k):
            v[i, j] = np.random.normal(0, 0.2)
    return w, v


# 定义误差损失函数 loss(y', y) = ∑-ln[sigmoid(y'* y)]
def get_loss(predict, y):
    x_shape = np.shape(predict)[0]
    loss = 0.0

    for i in range(x_shape):
        loss -= np.log(sigmoid(predict[i] * y[i]))

    return loss


# 梯度下降法优化模型参数
def stocGradient(dataMatrix, y, k, train_iter_num, alpha):
    """
        :param dataMatrix: 输入的数据集特征
        :param y: 特征对应的标签
        :param k: 交叉项矩阵的维度
        :param train_iter_num: 最大迭代次数
        :param alpha: 学习率
        :return:
    """
    m, n = np.shape(dataMatrix)
    w0 = 0
    w, v = initialize_w_v(n, k)
    # TODO regular
    # 训练次数
    for it in range(train_iter_num):
        # 对每个样本
        for x in range(m):
            v_1 = dataMatrix[x] * v
            v_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
            interaction = 0.5 * np.sum(np.multiply(v_1, v_1) - v_2)
            y_hat = w0 + dataMatrix[x] * w + interaction
            loss = sigmoid(y_hat * y[x]) - 1.0

            w0 -= alpha * loss * y[x]

            for i in range(n):
                if dataMatrix[x, i] != -1:
                    w[i, 0] -= alpha * loss * y[x] * dataMatrix[x, i]

                    for j in range(k):
                        v[i, j] -= alpha * loss * y[x] * (dataMatrix[x, i] * v_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

        if it % 100 == 0:
            predict_list = prediction(dataMatrix, w0, w, v)
            print("ITER NUM: {}, LOSS: {}".format(it, get_loss(predict_list, y)))

    return w0, w, v


def train():
    # load train data
    feature_list, label_list = load_data("./data/train_data.txt")

    # train
    w0, w, v = stocGradient(np.mat(feature_list), label_list, 8, 100000, 0.02)

    # predict
    predict_result = prediction(np.mat(feature_list), w0, w, v)
    print("ACC: ", get_accuracy(predict_result, label_list))

    # save model
    save_mode("./data/weights_FM", w0, w, v)

if __name__ == "__main__":
    start = time.time()
    train()
    print("TRAIN COST TIME: ", (time.time() - start))
