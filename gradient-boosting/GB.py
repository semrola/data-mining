__author__ = 'sandi'

import time
import datetime
import Orange
from Orange.classification import Learner, Model
import numpy as np
import Orange.regression
import Orange.evaluation
import Orange.classification
from sklearn.cross_validation import ShuffleSplit, cross_val_score, train_test_split
import scipy as sp
from timeconv import convert

def get_timestamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S - %d.%m.%Y')
    return st

def get_eye_matrix(X):
    num_of_classes = count_class(X)[0]
    return np.eye(num_of_classes)[X.astype(int)], num_of_classes

def sumtest(res):
    return np.sum(res, axis=1)

def count_class(x):
    if x.size == 0:
        return 0
    try:
        st = 0
        diff = []
        for item in x:
            if item not in diff:
                diff.append(item)
                st += 1
    except BaseException:
        st = 0
        diff = []
    return st, diff

class GradBoostRLearner(Learner):
    """Gradient Boosting for Regression."""

    def __init__(self, learner, n_estimators=10, epsilon=1e-5, loss="squared"):
        super().__init__()
        self.n_estimators = n_estimators
        self.learner = learner  # base learner
        self.name = "gb " + self.learner.name + " " + loss
        self.epsilon = epsilon
        losses = {"huber": self.grad_huber_loss,
                  "squared": self.grad_squared_loss,
                  "abs": self.grad_abs_loss,
                  "kldiv": self.kldiv,
                  "logloss": llfun,
                  "kldiv2": self.kldiv2}
        self.loss = losses[loss]
        self.cost = self.KL

    def grad_squared_loss(self, y, f):
        """Negative gradiant for squared loss."""
        return y - f

    def grad_abs_loss(self, y, f):
        """Negative gradient for absolute loss."""
        return np.sign(y - f)

    def grad_huber_loss(self, y, f, delta=0.5):
        """Negative gradient for Huber loss."""
        r0 = y - f
        r1 = delta * np.sign(r0)
        return np.vstack((r0, r1)).T[np.arange(y.shape[0]), (np.abs(r0)>delta).astype(int)]

    def kldiv(self, y, p):
        return p - y

    def kldiv2(self, y, p):
        return y - p

    def KLCost(self, y, p):
        eps = 1e-15
        return np.sum(y * np.log(y / p + eps))

    def KL(self, Q, P):
        return np.sum(P * np.log(P / (np.exp(Q)/np.sum(Q))))

    def grad_approx(self, y, f, e=1e-4):
        eps = 1e-15
        y = np.ravel(y)
        f = np.ravel(f)
        f = np.reshape(f, (1, len(f)))
        lst = []
        for eps in np.identity(len(f)) * e:
            a = self.softmax1(f+eps).ravel()
            b = self.softmax1(f-eps).ravel()
            lst.append((self.cost(y, a) - self.cost(y, b))/(2*e))
        return np.array(lst).ravel()
        #return (self.KLCost(y, f+eps) - self.KLCost(y, f-eps))/(2*e)

    def softmax1(self, f): #pk
        return np.divide(np.exp(f), np.vstack(np.sum(np.exp(f), axis=1)))

    def fit(self, X, Y, W=None):
        return self.fit_storage(Orange.data.Table(X, Y))

    def fit_storage(self, data):
        """Fitter. Learns a set of models for gradient boosting."""

        models = []
        X = data.X
        Y, self.num_classes = get_eye_matrix(data.Y)

        Fs = np.zeros((X.shape[0], self.num_classes))
        Ps = self.softmax1(Fs)

        grads = self.loss(Y, Ps)

        for i in range(self.n_estimators):
            model_per_class = []
            for j in range(Y.shape[1]):
                newdata = Orange.data.Table(data.X, grads[:, j])
                model = self.learner(newdata)
                tmp = model(newdata)
                Fs[:, j] += tmp
                Ps = self.softmax1(Fs)
                #grads[:, j] = self.loss(Y[:, j], Ps[:, j])
                grads = self.loss(Y, Ps)
                model_per_class.append(model)
            models.append(model_per_class)

        self.models = models
        return self

    def predict(self, X):
        models = self.models
        result = []

        for i in range(self.num_classes):
            ms = np.array([m[i](X) for m in models])
            suma = np.sum(ms, axis=0)
            result.append(suma)

        return self.softmax1(np.array(result).T)

    def get_params(self, deep = False):
        '''used for CV'''
        return {'n_estimators': self.n_estimators, 'learner': self.learner}

    def predict_proba(self, X):
        '''used for CV'''
        return self.predict(X)


def main():
    global vsak
    print(get_timestamp())
    start = time.time()

    test = True
    if test:
        vsak = 100
    else:
        vsak = 1

    X, Y, Xtest = read('iris')

    data = Orange.data.Table(X,Y)


    #ml = Orange.regression.MeanLearner()
    #rf = Orange.classification.SimpleRandomForestLearner(n_estimators=50)
    #lr = Orange.regression.LinearRegressionLearner()
    #gb_sq = GradBoostRLearner(stree, n_estimators=50, loss="squared")
    #gb_abs = GradBoostRLearner(stree, n_estimators=50, loss="abs")
    #gb_huber = GradBoostRLearner(stree, n_estimators=50, loss="huber")
    stree = Orange.classification.SimpleTreeLearner(max_depth=5)
    gb_kldiv = GradBoostRLearner(stree, n_estimators=200, loss="kldiv")

    #ml = Orange.classification.SoftmaxRegressionLearner()
    Y, _ = get_eye_matrix(Y)
    #model = gb_kldiv.fit_storage(data)
    #res = model.predict(X)
    #print(res.shape)

    #print('logloss', llfun(Y, res))

    #gradient check
    f = np.random.random((Y.shape[0], Y.shape[1]))
    print('f', f.shape)
    print('Y', Y.shape)
    a = gb_kldiv.grad_approx1(Y, f)
    b = gb_kldiv.loss(Y, f)
    print('gradient', np.sum((np.ravel(a) - np.ravel(b))**2))
    #return
    #gb_kldiv.grad_approx()

    cv = False
    if cv:
        bestN, bestD, bestMean = CV(X, data.Y, gb_kldiv)
    else:
        bestN = 400
        bestMean = 2
        bestD = 6
    #return

    predTime = time.time()
    stree = Orange.classification.SimpleTreeLearner(max_depth=bestD)
    learner = GradBoostRLearner(stree, n_estimators=bestN, loss="kldiv")
    model = learner.fit_storage(data)
    res = model.predict(Xtest)
    print('Prediction time', convert(time.time() - predTime))
    #print('kldiv', res)

    #learner = GradBoostRLearner(stree, n_estimators=bestN, loss="logloss")
    #model = learner.fit_storage(data)
    #res = model.predict(Xtest)
    #print('logloss', res)



    if test:
        pass
        #stest = sumtest(res)
        #print('sumtest1', stest, stest.shape)
        #print('sumtest2', np.sum(1 - stest))
        #print(res)
    else:
        write(res, 'Booster_ests-' + str(bestN) + 'cv-' + str(round(bestMean, 2)) + '_' + str(round(time.time() - start, 3)))


    #learners = [ml, lr, rf, gb_sq, gb_abs, gb_huber]
    #res = Orange.evaluation.CrossValidation(poly, learners, k=10)
    #print("\n".join("{:>30} {:.2f}".format(m.name, r) for m, r in zip(learners, Orange.evaluation.RMSE(res))))
    print('Total exc time', convert(time.time() - start))
    pass

def CV(X, Y, learner):
    start = time.time()
    cv_split = ShuffleSplit(Y.shape[0], n_iter=3, test_size=0.3, random_state=42)

    max_mean = 5
    best_estimators = 0
    best_mean = 0
    best_depth = 0

    ests = [500]
    depths = [5]
    for e in ests:
        for d in depths:
            print('Depth:', d)
            print('Estimators:', e)
            start1 = time.time()
            stree = Orange.classification.SimpleTreeLearner(max_depth=d)
            #stree1 = Orange.classification.RandomForestLearner(n_estimators=3, max_depth=d, max_leaf_nodes=10)
            ucenec = GradBoostRLearner(stree, n_estimators=e, loss="kldiv")
            #ucenec1 = GradBoostRLearner(stree, n_estimators=e, loss="kldiv2")
            sc = cross_val_score(ucenec, X, Y, cv=cv_split, scoring = 'log_loss', n_jobs=-1)
            #sc1 = cross_val_score(ucenec1, X, Y, cv=cv_split, scoring = 'log_loss', n_jobs=-1)
            mean = -np.mean(sc)
            #mean1 = -np.mean(sc1)
            if mean < max_mean:
                max_mean = mean
                best_estimators = e
                best_mean = mean
                best_depth = d
            print('score: ', mean)
            #print('score1: ', mean1)
            print('Time', convert(time.time() - start1))
            print(get_timestamp())

    print('Total CV time', convert(time.time() - start))
    return best_estimators, best_depth, best_mean

def CV2(X, Y, learner, k=5):
    scores = []
    for i in range(k):
        print("{}\rI=", i+1, end=' ')
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=1/k, random_state=i)
        preds = []
        model = learner.fit(Xtr, Ytr)
        y_pred = model.predict(Xte)
        preds.append(y_pred)
        preds = np.array(preds).T
        sc = llfun(Yte, preds)
        print(sc)
        scores.append(sc)
    print('score:', np.mean(scores))

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = np.sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll


def read(data):
    global vsak
    if data == 'real':
        #vsak = 1
        X = readTrain()
        X = np.array(X).astype(float)
        Y = X[:, -1].copy() #zadnji stolpec je y
        Y -= 1
        X = np.delete(X, -1, axis=1)    # izbrisemo zadnji stolpec
        X = np.delete(X, 0, axis=1)    #izbrisemo idje
        Xtest = readTest()
        Xtest = np.array(Xtest).astype(float)
        Xtest = np.delete(Xtest, 0, axis=1)    #izbrisemo idje
    elif data == 'iris':
        data = Orange.data.Table('iris')
        X = data.X
        Y = data.Y
        Xtest = X

    print('Reading done.')
    print(X.shape, Y.shape)
    return X, Y, Xtest


def readTrain():
    x = []
    file = 'train.csv'
    #file = 'data4_reduced.csv'
    i = 0
    with open(file, 'r') as f:
        for _ in range(1):
            next(f)
        for line in f:
            splitted = line.split(',')
            splitted[-1] = splitted[-1].split('_')[1].strip()
            if i % vsak == 0:
                x.append(splitted)
            i += 1
    return x

def readTest():
    x = []
    file = 'test.csv'
    i= 0
    with open(file, 'r') as f:
        for _ in range(1):
            next(f)
        for line in f:
            splitted = line.split(',')
            x.append(splitted)
    return x

def write(X, name='newFile'):
    with open('uniform.csv', 'r') as f:
        for line in f:
            cols = line
            break
    with open(name + '.csv', 'wt') as out:
        out.write(cols)
        id = 1
        for line in X:
            tmp = str(id)
            id += 1
            for col in line:
                tmp += ',' + str(col)
            out.write('%s\n' % tmp)
    print('Writing done.')

main()