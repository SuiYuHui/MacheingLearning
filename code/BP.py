import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def sigmoid(Input):
    return 1.0 / (1.0 + np.math.exp(-Input))


def GetAverage(Y, Y_result):
    E = 0
    for i in range(len(Y)):
        E += (Y_result[i] - Y[i]) ** 2
    return E / 2


class BP_Model:

    def __init__(self, m):  # m为每层的节点个数
        self.para = []
        self.thre = []
        self.Output = []
        self.Input = []
        for i in range(len(m) - 1):
            curr_para = np.random.uniform(0, 1, size=(m[i + 1], m[i]))
            curr_thre = np.random.rand(m[i + 1])
            self.Output.append([0.0] * m[i + 1])
            self.Input.append([0.0] * m[i])
            self.para.append(curr_para)
            self.thre.append(curr_thre)
        self.Input.append([0.0] * m[-1])

    def ClearIO(self):
        for i in range(len(self.Output)):
            for j in range(len(self.Output[i])):
                self.Output[i][j] = 0.0
                self.Input[i + 1][j] = 0.0
        for i in range(len(self.Input[0])):
            self.Input[0][i] = 0.0

    def GetResult(self, X):
        self.Input[0] = X.copy()
        for k in range(len(self.para)):
            v = self.para[k]
            for i in range(len(v)):
                for j in range(len(v[i])):
                    if k == 0:
                        self.Input[k + 1][i] += v[i][j] * self.Input[k][j]
                    else:
                        self.Input[k + 1][i] += v[i][j] * self.Output[k - 1][j]
                self.Output[k][i] = sigmoid(self.Input[k + 1][i] - self.thre[k][i])

    def GetG(self, y_ori, y):
        g = []
        for i in range(len(y)):
            g.append(y[i] * (1 - y[i]) * (y_ori[i] - y[i]))
        return g

    def GetDeltaPara(self, g):
        DeltaPara = [[] for j in range(len(self.para))]
        Yita = 0.1
        for i in range(len(self.para) - 1, 0, -1):
            para = self.para[i]
            DeltaPara[i] = [0] * len(para)
            e = []
            for h in range(len(para[0])):
                e.append(0)
                for j in range(len(para)):
                    DeltaPara[i][j] = Yita * g[j]  # * self.Output[i - 1][h]
                    e[h] += (para[j][h] * g[j])
                e[h] *= self.Output[i - 1][h] * (1 - self.Output[i - 1][h])
            g = e.copy()
        for h in range(len(self.para[0])):
            DeltaPara[0].append(Yita * g[h])  # * self.Input[0][i]
        return DeltaPara

    def UpdatePara(self, DeltaPara):
        for k in range(len(DeltaPara) - 1, 0, -1):
            para = self.para[k]
            para_d = DeltaPara[k]
            for i in range(len(para)):
                for j in range(len(para[0])):
                    self.para[k][i][j] += (para_d[i] * self.Output[k - 1][j])
                self.thre[k][i] -= para_d[i]
        for h in range(len(self.para[0])):
            for i in range(len(self.para[0][0])):
                self.para[0][h][i] += DeltaPara[0][h] * self.Input[0][i]
            self.thre[0][h] -= DeltaPara[0][h]

    def gradDscent(self, D):
        max_time = 500
        for i in range(max_time):
            E = 0
            for X, Y in D:
                self.GetResult(X)
                Y_result = self.Output[-1]
                E += GetAverage(Y, Y_result)
                g = self.GetG(Y, Y_result)
                DeltaPara = self.GetDeltaPara(g)
                self.UpdatePara(DeltaPara)
                self.ClearIO()
            if i % 50 == 0:
                print("当前误差为:", E)

    def Predict(self, D):
        Result = []
        for X, Y in D:
            self.GetResult(X)
            result = self.Output[-1]
            maxv = 0.0
            index = 0
            for i, v in enumerate(result):
                if v > maxv:
                    index = i
                    maxv = v
            Result.append(index)
            self.ClearIO()
        return Result


def EX_Y(y, n):
    Y = [[0] * n for j in range(len(y))]
    for i in range(len(y)):
        Y[i][int(y[i])] = 1
    return Y

def MakeM(cfmat, y_pred, y_test):
    wrong_data = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i] == 0:
            cfmat[0, 0] += 1
        elif y_pred[i] == y_test[i] == 1:
            cfmat[1, 1] += 1
        elif y_pred[i] == y_test[i] == 2:
            cfmat[2, 2] += 1
        elif y_pred[i] == 0:
            wrong_data += 1
            if y_test[i] == 1:
                cfmat[0][1] += 1
            elif y_test[i] == 2:
                cfmat[0][2] += 1
        elif y_pred[i] == 1:
            wrong_data += 1
            if y_test[i] == 0:
                cfmat[1][0] += 1
            elif y_test[i] == 2:
                cfmat[1][2] += 1
        elif y_pred[i] == 2:
            wrong_data += 1
            if y_test[i] == 0:
                cfmat[2][0] += 1
            elif y_test[i] == 1:
                cfmat[2][1] += 1
    return  wrong_data

def DataPro(filt_path, D, D_pre):
    dataset = np.loadtxt(filt_path, delimiter=",", encoding='utf-8')
    X = dataset[:, 1:5]  # X大写表示为矩阵
    y = dataset[:, 5]  # y小写表示为向量# 类别数据集8*1
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, train_size=0.75, random_state=0)  # 分成4组，测试比例为0.25，训练比例是0.75
    x_train, x_test, y_train, y_test = [], [], [], []
    for train_index, test_index in ss.split(X, y):
        x_train, x_test = X[train_index], X[test_index]  # 训练集对应的值
        y_train, y_test = y[train_index], y[test_index]  # 类别集对应的值
    x_train = x_train.tolist()
    x_test = x_test.tolist()
    y_train = EX_Y(y_train, 3)
    y_pre = EX_Y(y_test, 3)
    for i in range(len(x_train)):
        D.append([x_train[i], y_train[i]])
    for i in range(len(x_test)):
        D_pre.append([x_test[i], y_pre[i]])
    return y_test

if __name__ == '__main__':
    file_path = "iris.csv"
    D = []
    D_pre = []
    y_test = DataPro(file_path, D, D_pre)
    M = np.random.randint(6, 8, 2)
    M = list(M)
    M.insert(0, 4)
    M.append(3)
    bpIris = BP_Model(M)
    bpIris.gradDscent(D)

    y_pred = bpIris.Predict(D_pre)
    cfmat = np.zeros((3, 3))
    wrong_data = MakeM(cfmat, y_pred, y_test)


    print("混淆矩阵为:\n", cfmat)
    print("准确率为:", (len(y_pred) - wrong_data) / len(y_pred))
