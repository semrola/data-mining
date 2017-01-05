import Orange
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from scoring import pearson, NORM_STD
import time
from sklearn.cross_validation import train_test_split



class SelectByPermutationTest:
    def __init__(self, p=0.01, measure=lambda x, y: np.abs(pearsonr(x, y)), n_permutations=1):
        self.__dict__ = vars()

    def __call__(self, X, y1):
        y = np.copy(y1)
        null = []
        for i in range(self.n_permutations):
            np.random.shuffle(y)
            null.extend([self.measure(X[:, i], y)[0] for i in range(X.shape[1])])

        ss = np.sort(null)
        index = int((1-self.p)*len(null))
        threshold = ss[index]

        scores = [self.measure(X[:, i], y1)[0] for i in range(X.shape[1])]
        #topscores = [s for s in scores if s > threshold]
        data = []
        idxs = []
        for a, b in enumerate(scores):
            if b > threshold:
                data.append(X[:, a])
                idxs.append(a)

        return np.array(data).T, idxs


class RemoveConstant(Orange.preprocess.preprocess.Preprocess):
    def __call__(self, data):
        oks = np.min(data, axis=0) != np.max(data, axis=0)
        atts = [data.domain.attributes[i] for i, ok in enumerate(oks) if ok]
        domain = Orange.data.Domain(atts, data.domain.class_vars, data.domain.metas)
        return Orange.data.Table(domain, data)

def remove_constant(X):
    Xnew = []
    oks = np.min(X, axis=0) != np.max(X, axis=0)
    for i, j in enumerate(oks):
        if j:
            Xnew.append(X[:, i])
    return np.array(Xnew).T

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
                pred = col[j]
                if pred < 0:
                    pred = 0
                f.write('%d\t%s\t%f\n' % (cids_to_predict[i], col_names[j], pred))


def score(predicted, realscores):
    rint = pearson(predicted[:, 0], realscores[:, 0])
    rval = pearson(predicted[:, 1], realscores[:, 1])
    rdecall = [pearson(predicted[:, i], realscores[:, i]) for i in range(2, 21)]
    rdec = np.mean(rdecall)
    rs = np.array([rint, rval, rdec])
    zs = rs/NORM_STD
    return np.mean(zs)


def CV(X, Y, learner, k=5):
    scores = []
    for i in range(k):
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=1/k, random_state=i)
        preds = []
        for j in range(Ytr.shape[1]):   # za en stolpec
            y = Ytr[:, j]
            learner.fit(Xtr, y)
            y_pred = learner.predict(Xte)
            preds.append(y_pred)
        preds = np.array(preds).T
        sc = score(preds, Yte)
        scores.append(sc)
    print('score:', np.mean(scores))    # ali np.average() - to je sedaj vprasanje

def null_dist(X, y):
    pc_null = []
    for k in range(100):    # x-krat shufflamo
        np.random.shuffle(y)
        pc_null.extend([np.abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])])
    p = 0.01
    threshold = np.sort(pc_null)[(1-p)*len(pc_null)]


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

    # odstrani konstantne stolpce
    X = remove_constant(X)
    print('X no const shape', X.shape)

    predict = []
    with open(file4, 'r') as pred:
        for line in pred:
            cid = line.split('\t')[0]
            predict.append(float(cid))

    train_cids = []
    with open('TrainCids.txt', 'r') as cids:
        for cid in cids:
            train_cids.append(float(cid.strip()))
    #print('cids:', train_cids)


    #sel = VarianceThreshold(threshold=0.02)
    #X = sel.fit_transform(X)
    #print('reduced X shape', X.shape)

    # razdeli X na Xtrain in Xtest
    #Xtrain, Xtest = razdeli(X, predict, train_cids)

    # pridobi imena stolpcev
    with open(file1) as f1:
        paramNames = f1.readline().strip().split('\t')[6:]

    Xtrain, Xtest = razdeli(X, predict, train_cids)
    print('Xtrain shape', Xtrain.shape, end=" ")
    print('Xtest shape', Xtest.shape, end=" ")
    print('Y shape', Y.shape)

    method = 'ElasticNet'
    alpha = 0.05
    l1ratio = 0.6
    norm = True
    fit = True
    net = ElasticNet(alpha=alpha, l1_ratio=l1ratio, fit_intercept=True, normalize=True, precompute='auto', max_iter=10000)
    fss = SelectByPermutationTest(p=0.1, n_permutations=5)  # NULL_DIST
    nulldist = False
    cv = False
    start = time.time()
    if cv:
        ratios = [x/10 for x in range(1, 10)]
        alphas = [0.08, 0.07]
        for r in ratios:
            for a in alphas:
                print('a=',a,'r=',r, end=" ")
                net = ElasticNet(alpha=a, l1_ratio=r, fit_intercept=fit, normalize=norm, precompute='auto', max_iter=10000)
                CV(Xtrain, Y, net)
    else:
        predictions = []
        for i in range(Y.shape[1]):
            y = Y[:, i]
            if nulldist:
                Xtrain_new, cols = fss(Xtrain, y)
                print('SPT', Xtrain_new.shape, end=" ")
                xn = [Xtest[:, a] for a in cols]
                Xtest_new = np.array(xn).T
                print('SPT', Xtest_new.shape)
            else:
                Xtrain_new = Xtrain
                Xtest_new = Xtest
            params = []
            net.fit(Xtrain_new, y)
            params = net.predict(Xtest_new)

            #model = mean(Orange.data.Table(Xnew, y))
            #for ins in Xtest:
            #    params.append(model(ins))
            #print(params)
            predictions.append(params)


        fileName = 'mean_method=' + method + '_a=' + str(alpha) + '_l1=' + str(l1ratio) + '_norm=' + str(norm)
        write_result(fileName, predictions, paramNames, predict)

    print('Total time:', time.time()-start)


    '''
    lr = Orange.regression.LinearRegressionLearner()
    mean = Orange.regression.MeanLearner()
    ridge1 = Orange.regression.RidgeRegressionLearner(alpha = 10)
    ridge2 = Orange.regression.RidgeRegressionLearner(alpha = 1)
    ridge3 = Orange.regression.RidgeRegressionLearner(alpha = 0.5)
    ridge4 = Orange.regression.RidgeRegressionLearner(alpha = 0.1)
    lasso = Orange.regression.LassoRegressionLearner(alpha = 1)
    elastic = Orange.regression.ElasticNetLearner()
    elasticCV = Orange.regression.ElasticNetCVLearner()
    forest = Orange.classification.SimpleRandomForestLearner(n_estimators=10)
    #learners = [mean, ridge, lasso, elastic]
    learners = [ridge1,ridge2,ridge3,ridge4]

    print('RMSE...')
    rmse = []
    for i in range(Y.shape[1]):
        #break
        y = Y[:, i]
        data = Orange.data.Table(Xtrain, y)
        res = Orange.evaluation.CrossValidation(data, learners, k=5)
        #print(res)
        rmse.append(Orange.evaluation.RMSE(res))
        print(i, end="")

    rmse = np.array(rmse)
    avgs = np.average(rmse, axis=0)
    print(avgs)
    '''




main()



