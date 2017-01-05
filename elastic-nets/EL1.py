import Orange
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, RidgeCV
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
import time


class SelectByPermutationTest:
    def __init__(self, p=0.01, measure=lambda x, y: np.abs(pearsonr(x, y)), n_permutations=1):
        self.__dict__ = vars()

    def __call__(self, data):
        y = np.copy(data.Y)
        X = data.X
        null = []
        for i in range(self.n_permutations):
            np.random.shuffle(y)
            null.extend([self.measure(X[:, i], y)[0] for i in range(X.shape[1])])

        ss = np.sort(null)
        index = int((1-self.p)*len(null))
        threshold = ss[index]

        scores = [self.measure(X[:, i], data.Y)[0] for i in range(X.shape[1])]
        atts = [att for score, att in zip(scores, data.domain.attributes) if score > threshold]
        domain = Orange.data.Domain(atts, data.domain.class_vars, data.domain.metas)
        return Orange.data.Table(domain, data)


class RemoveConstant(Orange.preprocess.preprocess.Preprocess):
    def __call__(self, data):
        oks = np.min(data, axis=0) != np.max(data, axis=0)
        atts = [data.domain.attributes[i] for i, ok in enumerate(oks) if ok]
        domain = Orange.data.Domain(atts, data.domain.class_vars, data.domain.metas)
        return Orange.data.Table(domain, data)


def is_number(s):
    try:
        s = float(s)
        if np.isnan(s):
            return False
        else:
            return True
    except(ValueError, TypeError):
        return False


def razdeli(data, predict, train_cids):
    Xtrain = []
    Xtest = []
    #print(predict)
    for line in data:
        cid = line[0]
        tmp = line[1:]
        if cid in predict:
            Xtest.append(tmp)
        elif cid in train_cids:
            Xtrain.append(tmp)
    return np.array(Xtrain), np.array(Xtest)


def write_result(file, data, col_names, cids_to_predict):
    data = np.array(data)
    cids_to_predict.sort()
    with open(file+'.txt', 'w') as f:
        for i in range(data.shape[1]):
            col = data[:, i]
            for j in range(len(col)):
                f.write('%d\t%s\t%f\n' % (cids_to_predict[i], col_names[j], col[j]))


def main():
    file1 = 'TrainSet-hw2.txt'  # ucni podatki
    file2 = 'molecular_descriptors_data.txt'  # znacilnosti molekul sorted po CIDu
    file3 = 'train.txt'
    file4 = 'predict.txt'

    Y = []
    with open(file3, 'r') as train:
        for line in train:
            split = line.split('\t')
            tmp = []
            for x in split:
                tmp.append(x.strip())
            Y.append(tmp)

    Y = np.array(Y)
    Y = Y.astype(float)
    #print(Y)
    print(Y.shape)

    print('Reading X...')
    X = []
    with open(file2, 'r') as desc:
        for _ in range(1):  # skip first line
            next(desc)
        for line in desc:
            split = line.strip().split('\t')
            tmp = []
            for x in split:
                if not is_number(x):
                    tmp.append(0)
                    #print(x)
                else:
                    tmp.append(x)
            X.append(tmp)

    X = np.array(X)
    X = X.astype(float)
    print('X shape', X.shape)

    predict = []
    with open(file4, 'r') as pred:
        for line in pred:
            cid = line.split('\t')[0]
            predict.append(float(cid))

    train_cids = []
    with open('TrainCids.txt', 'r') as cids:
        for cid in cids:
            train_cids.append(float(cid.strip()))
    print('cids:', train_cids)

    # odstrani konstantne stolpce
    #sel = VarianceThreshold(threshold=0.02)
    #X = sel.fit_transform(X)
    #print('reduced X shape', X.shape)

    # razdeli X na Xtrain in Xtest
    Xtrain, Xtest = razdeli(X, predict, train_cids)

    # pridobi imena stolpcev
    with open(file1) as f1:
        paramNames = f1.readline().strip().split('\t')[6:]

    print('Model building...')
    elastic = Orange.regression.ElasticNetLearner()

    print('Xtrain shape', Xtrain.shape)
    print('Xtest shape', Xtest.shape)
    print('Y shape', Y.shape)

    start = time.time()
    # GRADNJA MODELA
    # obj = ElasticNet(alpha=1, l1_ratio=1, fit_intercept=True, normalize=False, max_iter=2500, positive=True)
    method = 'Elastic'
    alpha = 0.1
    l1ratio = 0.1
    normalize = True
    net = ElasticNet(alpha=alpha, l1_ratio=l1ratio, fit_intercept=True, normalize=True, precompute='auto')
    #net = RidgeCV()

    predictions = []
    for i in range(Y.shape[1]):
        y = Y[:, i]
        #print('y', y)
        #print('Fitting...')
        net.fit(Xtrain, y)
        #print('Predicting...')
        params = net.predict(Xtest)
        predictions.append(params)

    fileName = 'mean_method=' + method + '_a=' + str(alpha) + '_l1=' + str(l1ratio) + '_norm=' + str(normalize)
    write_result(fileName, predictions, paramNames, predict)

    print('Total time:', time.time()-start)


main()


