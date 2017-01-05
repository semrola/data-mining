__author__ = 'Sandi'
import Orange
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from scoring import pearson, NORM_STD
import time
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.decomposition import PCA


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

    try:
        data = data.T
        cids_to_predict.sort()
        with open(file+'.txt', 'w') as f:
            for i in range(data.shape[1]):
                col = data[:, i]
                for j in range(len(col)):
                    pred = col[j]
                    if pred < 0:
                        pred = 0
                    f.write('%d\t%s\t%f\n' % (cids_to_predict[i], col_names[j], pred))
    except BaseException as e:
        print(e)
        print(i, j)


def score(predicted, realscores):
    rint = pearson(predicted[:, 0], realscores[:, 0])
    rval = pearson(predicted[:, 1], realscores[:, 1])
    rdecall = [pearson(predicted[:, i], realscores[:, i]) for i in range(2, 21)]
    rdec = np.mean(rdecall)
    rs = np.array([rint, rval, rdec])
    zs = rs/NORM_STD
    return np.mean(zs)


def CV(X, Y, learner, k=6):
    scores = []
    for i in range(k):
        print("I=", i+1, end=' ')
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
    scs = np.mean(scores)   # ali np.average() - to je sedaj vprasanje
    print('score:', scs)
    return scs

def null_dist(X, y):
    pc_null = []
    for k in range(100):    # x-krat shufflamo
        np.random.shuffle(y)
        pc_null.extend([np.abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])])
    p = 0.01
    threshold = np.sort(pc_null)[(1-p)*len(pc_null)]


def makeXtrain(preds, i):
    a = np.array([])
    for x in range(len(preds)): #za vsak learner
        s = np.array([r[:, i] for r in preds[x]])   # za i-ti stolpec, x-ti learner
        s = s.flatten()
        if x == 0:
            a = np.hstack((a,s))
        else:
            a = np.vstack((a,s))
    return a.T


def stacking(learners, Xtrain, Xtest, Y, k=6):
    rows = Xtrain.shape[0]
    fold = int(rows / k)
    lastIndexOfTrain = fold*(k-1)

    learnerPredictions = [[] for _ in range(len(learners))] # seznam ki vsebuje 21 stolpicne arraye (kolikor je learnerjev)
    for i in range(k):
        Xtr = Xtrain[0:lastIndexOfTrain]
        Ytr = Y[0:lastIndexOfTrain]
        Xte = Xtrain[lastIndexOfTrain:]

        Xtrain = np.roll(Xtrain, fold, 0)
        Y = np.roll(Y, fold, 0)

        print("I =", i+1, end=' ')

        for idx, learner in enumerate(learners):
            foldpreds = []  # 21 stolcev napovedi za en learner, za en fold (1/k napovedi)
            for j in range(Ytr.shape[1]):   # po en stolpec
                y = Ytr[:, j]
                learner.fit(Xtr, y)
                y_pred = learner.predict(Xte)
                foldpreds.append(y_pred)
            foldpreds = np.array(foldpreds).T
            learnerPredictions[idx].append(foldpreds)
    for id in range(len(learnerPredictions)):
        learnerPredictions[id] = learnerPredictions[id][::-1]

    trainpreds = []
    for learn in learners:
        preds = []
        for m in range(Y.shape[1]):
            y = Y[:, m]
            learn.fit(Xtrain, y)
            y_pred = learn.predict(Xtest)
            preds.append(y_pred)
        preds = np.array(preds).T
        trainpreds.append(preds)

    try:
        preds = []
        rig = Ridge(alpha=5, normalize=True)
        for i in range(Y.shape[1]):
            y = Y[:, i]
            Xtr = makeXtrain(learnerPredictions, i)
            Xte = np.array([r[:, i] for r in trainpreds]).T
            #print('shapes:', y.shape, Xtr.shape, Xte.shape)
            rig.fit(Xtr, y)
            preds.append(rig.predict(Xte))
    except BaseException as e:
        print('EXC', e)
        print('y:', y)
        print('Xtr', Xtr)
        print('Xte', Xte)
        print('i', i)
        print('preds', preds)
        raise

    newY = np.array(preds).T
    #print('SHAPE', newY.shape)
    # CV(Xtrain, newY, rig)
    return newY

def main():
    file1 = 'TrainSet-hw2.txt'  # ucni podatki
    file2 = 'molecular_descriptors_data.txt'  # znacilnosti molekul sorted po CIDu
    file3 = 'train.txt'
    file4 = 'predict.txt'

    Y = []
    Y1k = []
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

    start = time.time()

    methods = ['forest', 'ridge','ridgePca','Lasso','LassoPca','ElasticNet']

    alphas = [0.05, 0.1, 0.4, 0.7, 1]
    pca = PCA()
    Xpca = pca.fit_transform(Xtrain)
    print('Xpca shape', Xpca.shape)

    '''
    evals = [[] for _ in range(len(methods))]
    learners = [[] for _ in range(len(methods))]
    # ocenjevanje posameznih metod
    forest = RandomForestClassifier(max_features='sqrt', n_jobs=-1, n_estimators=100)
    evals[0].append(CV(Xtrain, Y, forest))
    for alpha in alphas:
        print('Alpha:', alpha)
        ridge = Ridge(normalize=True, alpha=alpha)
        evals[1].append(CV(Xtrain, Y, ridge))
        evals[2].append(CV(Xpca, Y, ridge))
        las = Lasso(alpha=alpha, normalize=True)
        evals[3].append(CV(Xtrain, Y, las))
        evals[4].append(CV(Xpca, Y, las))
        net = ElasticNet(alpha=alpha, l1_ratio=0.6, normalize=True)
        evals[5].append(CV(Xtrain, Y, net))
    '''
    ridge = Ridge(normalize=True, alpha=1)
    las = Lasso(alpha=0.05, normalize=True)
    net = ElasticNet(alpha=0.05, l1_ratio=0.6, normalize=True)
    learners = [ridge, las, net]
    print('PAZI',len(paramNames), len(predict))
    newY = stacking(learners, Xtrain, Xtest, Y)
    write_result('haha', newY, paramNames, predict)

    return
    cv = True
    nulldist = False
    if cv:
        a = [x/100 for x in range(2, 10, 1)]
        #a = [0.0000001]
        #for k in a:
        for k in a:
            print('a=', k, end=" ")
            net = LogisticRegression(penalty ='l2', dual=False, fit_intercept=fit, C=k)
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



